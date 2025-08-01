# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class ReplayBufferData:
    def __init__(self, observations, actions, rewards, next_observations, dones):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones


class ReplayBuffer:
    def __init__(self, buffer_size, obs_space, action_space, device, n_envs):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.ptr = 0
        self.size = 0

        # Initialize storage buffers
        self.observations = obs_space.zeros(buffer_size, device=device)
        self.actions = action_space.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.next_observations = obs_space.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)

    def add(self, obs, next_obs, actions, rewards, dones, infos):
        # Calculate storage indices for all environments at once
        indices = (self.ptr + torch.arange(self.n_envs, device=self.observations.device)) % self.buffer_size

        # Vectorized storage - all environments at once
        self.observations[indices] = obs
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_observations[indices] = next_obs
        self.dones[indices] = dones

        # Update pointer and size
        self.ptr = (self.ptr + self.n_envs) % self.buffer_size
        self.size = min(self.size + self.n_envs, self.buffer_size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.observations.device)

        return ReplayBufferData(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices]
        )


class Space:
    def __init__(self, shape, dtype=np.float32, low=None, high=None):
        self.shape = shape
        self.dtype = dtype
        self.low = low
        self.high = high
        # Precalculate flat_dim since space is immutable
        if isinstance(self.shape, (list, tuple)):
            self.flat_dim = int(np.prod(self.shape))
        else:
            self.flat_dim = int(self.shape)

    def sample(self):
        """Sample a random value from this space."""
        if self.low is not None and self.high is not None:
            return np.random.uniform(self.low, self.high).astype(self.dtype)
        else:
            return np.random.randn(*self.shape).astype(self.dtype)

    def zeros(self, *batch_dims, device=None, dtype=None):
        """Create a zero tensor with this space's shape."""
        tensor_dtype = dtype or torch.float32
        if isinstance(self.shape, (list, tuple)):
            full_shape = batch_dims + self.shape
        else:
            full_shape = batch_dims + (self.shape,)
        return torch.zeros(full_shape, dtype=tensor_dtype, device=device)

    def ones(self, *batch_dims, device=None, dtype=None):
        """Create a ones tensor with this space's shape."""
        tensor_dtype = dtype or torch.float32
        if isinstance(self.shape, (list, tuple)):
            full_shape = batch_dims + self.shape
        else:
            full_shape = batch_dims + (self.shape,)
        return torch.ones(full_shape, dtype=tensor_dtype, device=device)

    def sample_tensor(self, *batch_dims, device=None, dtype=None):
        """Create a random tensor sampled from this space."""
        tensor_dtype = dtype or torch.float32
        if isinstance(self.shape, (list, tuple)):
            full_shape = batch_dims + self.shape
        else:
            full_shape = batch_dims + (self.shape,)

        if self.low is not None and self.high is not None:
            # Uniform sampling within bounds
            low_tensor = torch.tensor(self.low, dtype=tensor_dtype, device=device)
            high_tensor = torch.tensor(self.high, dtype=tensor_dtype, device=device)
            return torch.rand(full_shape, dtype=tensor_dtype, device=device) * (high_tensor - low_tensor) + low_tensor
        else:
            # Normal distribution for unbounded spaces
            return torch.randn(full_shape, dtype=tensor_dtype, device=device)


@dataclass
class Args:
    seed: int = 1
    total_timesteps: int = 1000000
    num_envs: int = 1
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default:0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    log_std_max: float = 2.0
    """maximum log standard deviation for policy"""
    log_std_min: float = -5.0
    """minimum log standard deviation for policy"""


class SoftQNetwork(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        num_observations = obs_space.flat_dim
        num_actions = action_space.flat_dim
        hidden_size = 256
        self.fc1 = nn.Linear(
            num_observations + num_actions,
            hidden_size,
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space):
        super().__init__()
        self.args = args
        num_observations = obs_space.flat_dim
        num_actions = action_space.flat_dim
        action_low = action_space.low
        action_high = action_space.high

        self.fc1 = nn.Linear(num_observations, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, num_actions)
        self.fc_logstd = nn.Linear(256, num_actions)

        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.args.log_std_min + 0.5 * (self.args.log_std_max - self.args.log_std_min) * (
                    log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SAC:
    """
    SAC (Soft Actor-Critic) algorithm implementation.
    """

    def __init__(self, args, obs_space, action_space, device):
        self.args = args
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = device

        # Initialize networks
        self.actor = Actor(args, obs_space, action_space).to(device)
        self.qf1 = SoftQNetwork(obs_space, action_space).to(device)
        self.qf2 = SoftQNetwork(obs_space, action_space).to(device)
        self.qf1_target = SoftQNetwork(obs_space, action_space).to(device)
        self.qf2_target = SoftQNetwork(obs_space, action_space).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Initialize optimizers
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=args.q_lr
        )
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)

        # Initialize entropy regularization
        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor([action_space.flat_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha = args.alpha
            self.a_optimizer = None

        self.rb = ReplayBuffer(
            self.args.buffer_size,
            self.obs_space,
            self.action_space,
            self.device,
            n_envs=self.args.num_envs,
        )

    def pre_step(self, global_step, obs):
        if global_step < self.args.learning_starts:
            actions = self.action_space.sample_tensor(self.args.num_envs, self.device)
        else:
            actions, _, _ = self.actor.get_action(obs)
        return actions

    def post_step(self, global_step, step_data):
        actions, next_obs, rewards, terminations, truncations, infos, obs = step_data

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        if global_step > self.args.learning_starts:
            data = self.rb.sample(self.args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
                qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
            qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            if global_step % self.args.policy_frequency == 0:
                for _ in range(self.args.policy_frequency):
                    pi, log_pi, _ = self.actor.get_action(data.observations)
                    qf1_pi = self.qf1(data.observations, pi)
                    qf2_pi = self.qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(data.observations)
                        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

            if global_step % self.args.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        return next_obs


def example_usage(envs):

    args = Args()
    run_name = f"PET__SAC__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps")

    # Create space objects
    obs_space = envs.obs_space
    action_space = envs.action_space

    # Reset environment externally
    initial_obs = envs.reset(seed=args.seed)

    # Create SAC algorithm
    sac = SAC(args, obs_space, action_space, device)

    sac.init()
    obs = initial_obs

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        actions, next_obs, rewards, terminations, truncations, infos = sac.pre_step(global_step, obs)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        step_data = (actions, next_obs, rewards, terminations, truncations, infos, obs)
        obs = sac.post_step(global_step, step_data, start_time)

        if global_step % 100 == 0:
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", sac.alpha, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

    writer.close()
