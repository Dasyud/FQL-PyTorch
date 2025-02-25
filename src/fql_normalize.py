import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Basic MLP module
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.GELU, activate_final=False, final_activation=nn.Tanh):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], output_dim))
        if activate_final is not False:
            layers.append(final_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------
# Critic module with ensemble support
# -------------------------------
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, ensemble_size=2):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.nets = nn.ModuleList(
            [MLP(obs_dim + action_dim, hidden_dims, 1, nn.GELU, activate_final=False, final_activation=nn.ReLU) for _ in range(ensemble_size)]
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        outputs = [net(x) for net in self.nets]  # each output shape: (batch, 1)
        # Concatenate along last dim so that output has shape (batch, ensemble_size)
        return torch.cat(outputs, dim=-1)

# -------------------------------
# Actor module used for both BC flow and one-step flow
# -------------------------------
class ActorFlow(nn.Module):
    def __init__(self, obs_dim, input_dim, hidden_dims, output_dim):
        """
        For actor_bc_flow, input_dim = obs_dim + action_dim + 1 (with t).
        For actor_onestep_flow, input_dim = obs_dim + action_dim (with noise).
        """
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, output_dim, nn.Tanh, activate_final=False, final_activation=nn.Tanh)
    def forward(self, *args):
        # Concatenate all inputs along the last dimension.
        x = torch.cat(args, dim=-1)
        return self.net(x)

# -------------------------------
# Combined Network holding critic and actor modules
# -------------------------------
class FlowQLNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, ensemble_size=2, state_mean=None, state_std=None):
        super().__init__()
        # Critic and target critic networks
        self.critic = Critic(obs_dim, action_dim, hidden_dims, ensemble_size)
        self.target_critic = copy.deepcopy(self.critic)
        # Actor network for BC flow: takes [obs, candidate action, t]
        self.actor_bc_flow = ActorFlow(obs_dim, obs_dim + action_dim + 1, hidden_dims, action_dim)
        # Actor network for one-step policy: takes [obs, noise]
        self.actor_onestep_flow = ActorFlow(obs_dim, obs_dim + action_dim, hidden_dims, action_dim)
        # (Optional encoder for state-dependent std in actor can be added here.)
        # Register state normalization statistics
        if state_mean is not None and state_std is not None:
            # Make sure these are column vectors (or have shape [1, obs_dim]) so they broadcast with input [batch, obs_dim]
            self.register_buffer('state_mean', state_mean.unsqueeze(0))
            self.register_buffer('state_std', state_std.unsqueeze(0))
        else:
            self.state_mean = None
            self.state_std = None
    
    def update_target(self, tau):
        """
        Update target critic with exponential moving average.
        """
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# -------------------------------
# Flow Q-Learning Agent in PyTorch
# -------------------------------
class FlowQLAgent(nn.Module):
    def __init__(self, network, config):
        super().__init__()
        self.network = network
        self.config = config  # Expected keys: discount, tau, q_agg, alpha, flow_steps, normalize_q_loss, action_dim

    def _normalize_obs(self, obs):
        # Apply normalization if buffers exist.
        if self.network.state_mean is not None and self.network.state_std is not None:
            # obs = (obs - self.network.state_mean) / (self.network.state_std + 1e-6)
            obs = (obs - self.network.state_mean) / (self.network.state_std)
        return obs

    def _unnormalize_obs(self, obs):
        # Apply un-normalization if buffers exist.
        if self.network.state_mean is not None and self.network.state_std is not None:
            obs = obs * self.network.state_std + self.network.state_mean
        return obs

    def critic_loss(self, batch):
        """
        Compute the critic loss.
        batch is expected to be a dictionary with keys:
          'observations', 'actions', 'next_observations', 'rewards', 'masks'
        """
        obs = self._normalize_obs(batch['observations'])
        actions = batch['actions']
        next_obs = self._normalize_obs(batch['next_observations'])
        rewards = batch['rewards']
        masks = batch['masks']  # Typically 0/1 for terminal or non-terminal
        
        with torch.no_grad():
            # next_actions = self.sample_actions(self._unnormalize_obs(next_obs))
            next_actions = self.sample_actions(batch['next_observations'])
            next_actions = torch.clamp(next_actions, -1, 1)
            next_qs = self.network.target_critic(next_obs, next_actions)  # Shape: (batch, ensemble_size)
            if self.config['q_agg'] == 'min':
                next_q, _ = torch.min(next_qs, dim=1, keepdim=True)
            else:
                next_q = torch.mean(next_qs, dim=1, keepdim=True)
            target_q = rewards + self.config['discount'] * masks * next_q

        qs = self.network.critic(obs, actions)  # Shape: (batch, ensemble_size)
        critic_loss = F.mse_loss(qs, target_q.expand_as(qs))
        info = {
            'critic_loss': critic_loss.item(),
            'q_mean': qs.mean().item(),
            'q_max': qs.max().item(),
            'q_min': qs.min().item(),
        }
        return critic_loss, info

    def actor_loss(self, batch):
        """
        Compute the actor loss which includes:
         - A behavioral cloning flow loss (BC flow loss)
         - A distillation loss between one-step and computed flow actions
         - A Q-loss (to maximize Q, expressed as a negative term)
        """
        obs = self._normalize_obs(batch['observations'])
        true_actions = batch['actions']
        batch_size, action_dim = true_actions.shape
        device = obs.device

        # BC flow loss: sample initial points and interpolate
        x0 = torch.randn(batch_size, action_dim, device=device)
        x1 = true_actions
        t = torch.rand(batch_size, 1, device=device)  # Uniformly sampled t in [0, 1]
        x_t = (1 - t) * x0 + t * x1
        vel = x1 - x0

        # Compute predicted velocities from the BC flow network.
        actor_bc_input = torch.cat([obs, x_t, t], dim=-1)
        pred = self.network.actor_bc_flow(actor_bc_input)
        bc_flow_loss = F.mse_loss(pred, vel)

        # Distillation loss: align one-step flow with the computed multi-step flow.
        noises = torch.randn(batch_size, action_dim, device=device)
        # target_flow_actions = self.compute_flow_actions(self._unnormalize_obs(obs), noises)
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises)
        actor_flow_input = torch.cat([obs, noises], dim=-1)
        actor_actions = self.network.actor_onestep_flow(actor_flow_input)
        distill_loss = F.mse_loss(actor_actions, target_flow_actions)

        # Q loss: encourage actions to have high Q value.
        actor_actions_clipped = torch.clamp(actor_actions, -1, 1)
        qs = self.network.critic(obs, actor_actions_clipped).detach()  # (batch, ensemble_size)
        q = torch.mean(qs, dim=1, keepdim=True)
        q_loss = -q.mean()
        if self.config.get('normalize_q_loss', False):
            lam = 1.0 / (q.abs().mean().detach() + 1e-6)
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss
        mse = F.mse_loss(actor_actions, true_actions)
        info = {
            'actor_loss': actor_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            'distill_loss': distill_loss.item(),
            'q_loss': q_loss.item(),
            'q': q.mean().item(),
            'mse': mse.item(),
        }
        return actor_loss, info

    def total_loss(self, batch):
        """
        Compute the total loss as the sum of critic and actor losses.
        Returns both the loss and a dictionary of (logged) metrics.
        """
        critic_loss_val, critic_info = self.critic_loss(batch)
        actor_loss_val, actor_info = self.actor_loss(batch)
        loss = critic_loss_val + actor_loss_val
        info = {}
        for k, v in critic_info.items():
            info['critic/' + k] = v
        for k, v in actor_info.items():
            info['actor/' + k] = v
        return loss, info

    def target_update(self):
        """
        Update the target network for the critic using exponential moving average.
        """
        self.network.update_target(self.config['tau'])

    def sample_actions(self, observations):
        """
        Sample actions from the one-step flow actor.
        """
        observations = self._normalize_obs(observations)
        batch_size = observations.size(0)
        action_dim = self.config['action_dim']
        noises = torch.randn(batch_size, action_dim, device=observations.device)
        actor_flow_input = torch.cat([observations, noises], dim=-1)
        actions = self.network.actor_onestep_flow(actor_flow_input)
        return torch.clamp(actions, -1, 1)

    def compute_flow_actions(self, observations, noises):
        """
        Compute actions from the BC flow actor using the Euler method over the specified number of flow steps.
        """
        observations = self._normalize_obs(observations)
        batch_size, action_dim = noises.shape
        flow_steps = self.config['flow_steps']
        actions = noises  # Initialize with the sampled noises.
        for i in range(flow_steps):
            t_val = torch.full((batch_size, 1), i / flow_steps, device=observations.device)
            actor_input = torch.cat([observations, actions, t_val], dim=-1)
            vels = self.network.actor_bc_flow(actor_input)
            actions = actions + vels / flow_steps
            actions = torch.clamp(actions, -1, 1)
        return actions

    def update(self, batch, optimizer):
        """
        Perform a gradient update for the agent on a given batch.
        Returns a dictionary of metrics.
        """
        loss, info = self.total_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return info

# -------------------------------
# Utility: Configuration and Agent Creation
# -------------------------------
def get_config():
    """
    Return a configuration dictionary.
    You can modify these parameters for your environment.
    """
    config = {
        'agent_name': 'fql',
        'discount': 0.99,
        'tau': 0.005,
        'q_agg': 'mean',  # Use 'min' to aggregate critics via the minimum operator.
        'alpha': 10.0,    # Coefficient for the distillation loss.
        'flow_steps': 10,
        'normalize_q_loss': False,
        # action_dim and obs_dim will be set during agent creation.
    }
    return config

def create_fql_agent(seed, ex_observations, ex_actions, config, state_mean, state_std):
    """
    Example creation routine for the FQL agent.
    'ex_observations' and 'ex_actions' should be example tensors (from a batch) to infer dimensions.
    """
    torch.manual_seed(seed)
    obs_dim = ex_observations.shape[-1]
    action_dim = ex_actions.shape[-1]
    config['action_dim'] = action_dim
    hidden_dims = [512, 512, 512, 512]
    # hidden_dims = [128, 128]
    ensemble_size = 2  # Number of critic ensemble members.
    # Ensure state_mean and state_std are torch tensors.
    if not torch.is_tensor(state_mean):
        state_mean = torch.tensor(state_mean, dtype=torch.float32)
    if not torch.is_tensor(state_std):
        state_std = torch.tensor(state_std, dtype=torch.float32)
    network = FlowQLNetwork(obs_dim, action_dim, hidden_dims, ensemble_size,
                             state_mean=state_mean, state_std=state_std)
    agent = FlowQLAgent(network, config)
    return agent

# -------------------------------
# Example usage (to be integrated in your training loop)
# -------------------------------
if __name__ == '__main__':
    # Dummy example data for creating the agent
    ex_obs = torch.randn(10, 150)  # e.g., batch of 10, observation dimension 150
    ex_actions = torch.randn(10, 1)  # e.g., batch of 10, action dimension 1
    config = get_config()
    agent = create_fql_agent(seed=42, ex_observations=ex_obs, ex_actions=ex_actions, config=config)

    # Create an optimizer for the agent's parameters
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    # Dummy training batch (make sure dimensions match your actual data)
    batch = {
        'observations': torch.randn(32, 150),
        'actions': torch.randn(32, 1),
        'next_observations': torch.randn(32, 150),
        'rewards': torch.randn(32, 1),
        'masks': torch.ones(32, 1)  # 1 for non-terminal, 0 for terminal
    }

    # Perform one update step and then update target network
    info = agent.update(batch, optimizer)
    agent.target_update()

    print("Update info:", info)
