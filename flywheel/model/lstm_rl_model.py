#!/usr/bin/env python3
"""
LSTM-based Actor-Critic model for flywheel control using PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math

class LSTMActorCritic(nn.Module):
    """
    LSTM-based Actor-Critic network for continuous control.
    
    Architecture:
    - Shared LSTM backbone for processing sequential state information
    - Actor head: outputs action mean and log_std for continuous control
    - Critic head: outputs state value estimate
    """
    
    def __init__(
        self,
        state_dim: int = 6,  # [time, pwm, current, voltage, position, servo_enabled]
        action_dim: int = 20,  # [servo_position, servo_enable]
        hidden_size: int = 1280,
        num_layers: int = 2,
        dropout: float = 0.1,
        action_std_init: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(state_dim, hidden_size)
        
        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Action standard deviation (learnable parameter)
        self.action_log_std = nn.Parameter(
            torch.ones(action_dim) * math.log(action_std_init)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weight initialization
                    nn.init.xavier_uniform_(param)
                else:
                    # Linear layer weight initialization
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial hidden state for LSTM"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h_0, c_0
    
    def forward(
        self,
        states: torch.Tensor,  # (batch_size, sequence_length, state_dim)
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Returns:
            action_mean: Mean of action distribution (batch_size, sequence_length, action_dim)
            values: State values (batch_size, sequence_length, 1)
            new_hidden_state: Updated LSTM hidden state
        """
        batch_size, seq_len = states.shape[:2]
        
        # Project inputs
        x = self.input_proj(states)  # (batch, seq_len, hidden_size)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.get_initial_hidden_state(batch_size, states.device)
        
        # LSTM forward pass
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)  # (batch, seq_len, hidden_size)
        
        # Actor and critic outputs
        action_mean = self.actor_head(lstm_out)  # (batch, seq_len, action_dim)
        values = self.critic_head(lstm_out)      # (batch, seq_len, 1)
        
        # Apply different activations for different action dimensions
        # Servo position: tanh for [-1, 1] range (will be scaled to [0, 180])
        # Servo enable: sigmoid for [0, 1] range
        action_mean_pos = torch.tanh(action_mean[..., :1])  # First dimension: servo position
        action_mean_enable = torch.sigmoid(action_mean[..., 1:])  # Second dimension: servo enable
        action_mean = torch.cat([action_mean_pos, action_mean_enable], dim=-1)
        
        return action_mean, values, new_hidden_state
    
    def get_action_and_value(
        self,
        states: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action and value for given states.
        
        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of actions
            values: State values
            new_hidden_state: Updated LSTM hidden state
        """
        action_mean, values, new_hidden_state = self.forward(states, hidden_state)
        
        # Action standard deviation
        action_std = torch.exp(self.action_log_std)
        
        # Create action distribution
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            actions = action_mean
        else:
            actions = action_dist.sample()
        
        log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        return actions, log_probs, values, new_hidden_state
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states (used in PPO updates).
        
        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Action distribution entropy
        """
        action_mean, values, _ = self.forward(states, hidden_state)
        
        # Action standard deviation
        action_std = torch.exp(self.action_log_std)
        
        # Create action distribution
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, values, entropy

class PPOBuffer:
    """
    Buffer for storing PPO training data with proper handling of LSTM sequences.
    """
    
    def __init__(self, max_size: int, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Storage arrays
        self.states = torch.zeros((max_size, state_dim), device=device)
        self.actions = torch.zeros((max_size, action_dim), device=device)
        self.log_probs = torch.zeros((max_size, 1), device=device)
        self.values = torch.zeros((max_size, 1), device=device)
        self.rewards = torch.zeros((max_size, 1), device=device)
        self.dones = torch.zeros((max_size, 1), device=device, dtype=torch.bool)
        self.advantages = torch.zeros((max_size, 1), device=device)
        self.returns = torch.zeros((max_size, 1), device=device)
        
        # Episode boundaries for LSTM training
        self.episode_starts = []
        self.episode_lengths = []
        
        self.ptr = 0
        self.size = 0
        self.current_episode_start = 0
    
    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool
    ):
        """Store one step of experience"""
        if self.ptr >= self.max_size:
            return  # Buffer is full
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)
        
        # Track episode boundaries
        if done:
            episode_length = self.ptr - self.current_episode_start
            self.episode_starts.append(self.current_episode_start)
            self.episode_lengths.append(episode_length)
            self.current_episode_start = self.ptr
    
    def compute_advantages_and_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute advantages and returns using GAE"""
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        # Process each episode separately
        for start, length in zip(self.episode_starts, self.episode_lengths):
            end = start + length
            
            # Compute advantages for this episode
            last_gae_lam = 0
            for t in reversed(range(start, end)):
                if t == end - 1:
                    next_non_terminal = 0.0
                    next_value = 0.0
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1].float()
                    next_value = self.values[t + 1]
                
                delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
                advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
            # Compute returns
            returns[start:end] = advantages[start:end] + self.values[start:end]
        
        self.advantages = advantages
        self.returns = returns
        
        # Normalize advantages
        if self.size > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def get_batch_sequences(self, batch_size: int, max_seq_len: int = 32):
        """
        Get batches of sequences for LSTM training.
        
        Returns sequences of variable length up to max_seq_len, properly handling
        episode boundaries.
        """
        if self.size == 0:
            return None
        
        batches = []
        indices = list(range(self.size))
        np.random.shuffle(indices)
        
        i = 0
        while i < len(indices):
            batch_states = []
            batch_actions = []
            batch_log_probs = []
            batch_values = []
            batch_advantages = []
            batch_returns = []
            batch_lengths = []
            
            current_batch_size = 0
            
            while current_batch_size < batch_size and i < len(indices):
                start_idx = indices[i]
                
                # Find the episode this index belongs to
                episode_start = 0
                episode_end = self.size
                for ep_start, ep_len in zip(self.episode_starts, self.episode_lengths):
                    if ep_start <= start_idx < ep_start + ep_len:
                        episode_start = ep_start
                        episode_end = ep_start + ep_len
                        break
                
                # Create sequence starting from start_idx
                seq_len = min(max_seq_len, episode_end - start_idx)
                end_idx = start_idx + seq_len
                
                batch_states.append(self.states[start_idx:end_idx])
                batch_actions.append(self.actions[start_idx:end_idx])
                batch_log_probs.append(self.log_probs[start_idx:end_idx])
                batch_values.append(self.values[start_idx:end_idx])
                batch_advantages.append(self.advantages[start_idx:end_idx])
                batch_returns.append(self.returns[start_idx:end_idx])
                batch_lengths.append(seq_len)
                
                current_batch_size += 1
                i += 1
            
            if batch_states:
                batches.append({
                    'states': batch_states,
                    'actions': batch_actions,
                    'log_probs': batch_log_probs,
                    'values': batch_values,
                    'advantages': batch_advantages,
                    'returns': batch_returns,
                    'lengths': batch_lengths
                })
        
        return batches
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0
        self.current_episode_start = 0
        self.episode_starts.clear()
        self.episode_lengths.clear()

class PPOTrainer:
    """
    PPO trainer for LSTM Actor-Critic model.
    """
    
    def __init__(
        self,
        model: LSTMActorCritic,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def update(self, buffer: PPOBuffer, n_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """Update the model using PPO"""
        
        # Compute advantages and returns
        buffer.compute_advantages_and_returns()
        
        # Get batch sequences
        batches = buffer.get_batch_sequences(batch_size)
        if not batches:
            return {}
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0
        
        for epoch in range(n_epochs):
            for batch in batches:
                # Pad sequences to same length
                max_len = max(batch['lengths'])
                padded_states = []
                padded_actions = []
                padded_advantages = []
                padded_returns = []
                old_log_probs = []
                
                for i, length in enumerate(batch['lengths']):
                    # Pad with zeros
                    if length < max_len:
                        pad_len = max_len - length
                        states = torch.cat([
                            batch['states'][i],
                            torch.zeros(pad_len, self.model.state_dim, device=self.device)
                        ], dim=0)
                        actions = torch.cat([
                            batch['actions'][i],
                            torch.zeros(pad_len, self.model.action_dim, device=self.device)
                        ], dim=0)
                        advantages = torch.cat([
                            batch['advantages'][i],
                            torch.zeros(pad_len, 1, device=self.device)
                        ], dim=0)
                        returns = torch.cat([
                            batch['returns'][i],
                            torch.zeros(pad_len, 1, device=self.device)
                        ], dim=0)
                        log_probs = torch.cat([
                            batch['log_probs'][i],
                            torch.zeros(pad_len, 1, device=self.device)
                        ], dim=0)
                    else:
                        states = batch['states'][i]
                        actions = batch['actions'][i]
                        advantages = batch['advantages'][i]
                        returns = batch['returns'][i]
                        log_probs = batch['log_probs'][i]
                    
                    padded_states.append(states)
                    padded_actions.append(actions)
                    padded_advantages.append(advantages)
                    padded_returns.append(returns)
                    old_log_probs.append(log_probs)
                
                # Stack into batch tensors
                states_batch = torch.stack(padded_states)  # (batch_size, max_len, state_dim)
                actions_batch = torch.stack(padded_actions)
                advantages_batch = torch.stack(padded_advantages)
                returns_batch = torch.stack(padded_returns)
                old_log_probs_batch = torch.stack(old_log_probs)
                
                # Create attention mask for variable length sequences
                lengths_tensor = torch.tensor(batch['lengths'], device=self.device)
                mask = torch.arange(max_len, device=self.device)[None, :] < lengths_tensor[:, None]
                mask = mask.unsqueeze(-1).float()  # (batch_size, max_len, 1)
                
                # Forward pass
                new_log_probs, values, entropy = self.model.evaluate_actions(states_batch, actions_batch)
                
                # Apply mask to ignore padded timesteps
                new_log_probs = new_log_probs * mask
                values = values * mask
                entropy = entropy * mask
                old_log_probs_batch = old_log_probs_batch * mask
                advantages_batch = advantages_batch * mask
                returns_batch = returns_batch * mask
                
                # Compute ratio
                log_ratio = new_log_probs - old_log_probs_batch
                ratio = torch.exp(log_ratio)
                
                # Policy loss (PPO clipped objective)
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum() / mask.sum()
                
                # Value loss
                value_loss = F.mse_loss(values.sum() / mask.sum(), returns_batch.sum() / mask.sum(), reduction='mean')
                
                # Entropy loss (encourage exploration)
                entropy_loss = -entropy.sum() / mask.sum()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1
        
        return {
            'total_loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates
        }

# Example usage
if __name__ == "__main__":
    # Test model creation
    model = LSTMActorCritic(
        state_dim=6,  # Updated for 6D state space
        action_dim=2,
        hidden_size=128,
        num_layers=2
    )
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    states = torch.randn(batch_size, seq_len, 6)  # Updated for 6D state space
    
    action_mean, values, hidden_state = model(states)
    print(f"Action mean shape: {action_mean.shape}")
    print(f"Values shape: {values.shape}")
    print("LSTM Actor-Critic test passed!")
    
    # Test buffer
    buffer = PPOBuffer(max_size=1000, state_dim=6, action_dim=2)  # Updated for 6D state space
    print("PPO Buffer test passed!")