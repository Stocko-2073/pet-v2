#!/usr/bin/env python3
"""
Decision Transformer implementation for flywheel control.
Based on "Decision Transformer: Reinforcement Learning via Sequence Modeling"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DecisionTransformer(nn.Module):
    """
    Decision Transformer for continuous control of flywheel system.
    
    Input sequence: [s_0, a_0, r_0, s_1, a_1, r_1, ..., s_t]
    Output: a_t (next action)
    """
    
    def __init__(
        self,
        state_dim: int = 6,           # Flywheel state dimension [time, pwm, current, voltage, position, servo_enabled]
        action_dim: int = 2,          # [servo_position, servo_enable]
        hidden_size: int = 128,       # Transformer hidden size
        max_length: int = 20,         # Maximum sequence length
        n_layer: int = 3,             # Number of transformer layers
        n_head: int = 4,              # Number of attention heads
        activation: str = 'relu',     # Activation function
        dropout: float = 0.1,         # Dropout rate
        action_tanh: bool = True,     # Whether to use tanh for action output
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.action_tanh = action_tanh
        
        # Token embedding layers
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(action_dim, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)  # Return-to-go
        
        # Positional encoding (increased size for [r,s,a] triplets)
        self.pos_encoding = PositionalEncoding(hidden_size, max_length * 3)
        
        # Transformer
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layer)
        
        # Action prediction head
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
    def forward(
        self,
        states: torch.Tensor,           # (batch, seq_len, state_dim)
        actions: torch.Tensor,          # (batch, seq_len, action_dim)
        returns_to_go: torch.Tensor,    # (batch, seq_len, 1)
        timesteps: torch.Tensor,        # (batch, seq_len)
        attention_mask: Optional[torch.Tensor] = None
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=states.device)
        
        # Embed tokens
        state_embeddings = self.embed_state(states)      # (batch, seq_len, hidden)
        action_embeddings = self.embed_action(actions)   # (batch, seq_len, hidden)
        return_embeddings = self.embed_return(returns_to_go)  # (batch, seq_len, hidden)
        
        # Create sequence: [r_0, s_0, a_0, r_1, s_1, a_1, ...]
        # For each timestep t, we use [r_t, s_t] to predict action a_t
        sequence_embeddings = []
        
        for t in range(seq_length):
            sequence_embeddings.append(return_embeddings[:, t])  # r_t
            sequence_embeddings.append(state_embeddings[:, t])   # s_t
            sequence_embeddings.append(action_embeddings[:, t])  # a_t (include all actions for consistent length)
        
        # Stack embeddings: (batch, seq_len*3, hidden)
        stacked_inputs = torch.stack(sequence_embeddings, dim=1)
        
        # Add positional encoding and normalize
        stacked_inputs = self.pos_encoding(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Apply transformer (now batch_first=True)
        transformer_outputs = self.transformer(stacked_inputs)  # (batch, seq_len*3, hidden)
        
        # Extract state representations for action prediction
        # We want the state embeddings: indices 1, 4, 7, ... (every 3rd, starting from 1)
        # These correspond to the state positions in [r, s, a, r, s, a, ...]
        state_indices = torch.arange(1, transformer_outputs.size(1), 3, device=transformer_outputs.device)
        state_outputs = transformer_outputs[:, state_indices, :]  # (batch, seq_len, hidden)
        
        # Predict actions
        action_preds = self.predict_action(state_outputs)  # (batch, seq_len, action_dim)
        
        if self.action_tanh:
            # Apply different activations for different action dimensions
            # Servo position: tanh for [-1, 1] range (will be scaled to [0, 180])
            # Servo enable: sigmoid for [0, 1] range
            action_pos = torch.tanh(action_preds[..., :1])  # First dimension: servo position
            action_enable = torch.sigmoid(action_preds[..., 1:])  # Second dimension: servo enable
            action_preds = torch.cat([action_pos, action_enable], dim=-1)
            
        return action_preds
    
    def get_action(
        self,
        states: torch.Tensor,           # (seq_len, state_dim)
        actions: torch.Tensor,          # (seq_len, action_dim)
        returns_to_go: torch.Tensor,    # (seq_len, 1)
        timesteps: torch.Tensor,        # (seq_len,)
    ):
        """Get single action for inference"""
        # Add batch dimension
        states = states.unsqueeze(0)
        actions = actions.unsqueeze(0)
        returns_to_go = returns_to_go.unsqueeze(0)
        timesteps = timesteps.unsqueeze(0)
        
        with torch.no_grad():
            action_preds = self.forward(states, actions, returns_to_go, timesteps)
            # Return last predicted action
            return action_preds[0, -1]  # (action_dim,)

class DecisionTransformerTrainer:
    """Trainer for Decision Transformer"""
    
    def __init__(
        self,
        model: DecisionTransformer,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        target_actions: torch.Tensor
    ):
        """Single training step"""
        self.model.train()
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns_to_go = returns_to_go.to(self.device)
        timesteps = timesteps.to(self.device)
        target_actions = target_actions.to(self.device)
        
        # Forward pass
        action_preds = self.model(states, actions, returns_to_go, timesteps)
        
        # Compute loss (MSE for continuous actions)
        loss = F.mse_loss(action_preds, target_actions)
        
        # Debug: Print loss components occasionally
        if hasattr(self, 'step_count'):
            self.step_count += 1
        else:
            self.step_count = 1
            
        if self.step_count % 100 == 0:  # Print every 100 steps
            pred_range = f"[{action_preds.min().item():.3f}, {action_preds.max().item():.3f}]"
            target_range = f"[{target_actions.min().item():.3f}, {target_actions.max().item():.3f}]"
            print(f"Step {self.step_count}: Loss={loss.item():.6f}, Pred range={pred_range}, Target range={target_range}")
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

class FlywheelController:
    """Real-time controller using Decision Transformer"""
    
    def __init__(
        self,
        model_path: str,
        context_length: int = 20,
        device: str = 'cpu',
        action_scale: float = 90.0,  # Scale tanh output to 0-180 degrees
        action_offset: float = 90.0
    ):
        self.device = device
        self.context_length = context_length
        self.action_scale = action_scale
        self.action_offset = action_offset
        
        # Load model
        self.model = DecisionTransformer()
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        
        # History buffers
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.timestep = 0
        
    def reset(self):
        """Reset controller history"""
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.timestep = 0
        
    def get_action(
        self, 
        state: np.ndarray, 
        reward: float = 0.0,
        target_return: float = 1.0
    ) -> float:
        """
        Get next action given current state
        
        Args:
            state: Current state (5-dim numpy array)
            reward: Reward from last step
            target_return: Desired return-to-go
            
        Returns:
            Servo position (0-180 degrees)
        """
        # Add to history
        self.state_history.append(state.copy())
        if len(self.reward_history) > 0 or reward != 0.0:
            self.reward_history.append(reward)
        
        # Truncate history to context length
        if len(self.state_history) > self.context_length:
            self.state_history = self.state_history[-self.context_length:]
            self.action_history = self.action_history[-self.context_length:]
            self.reward_history = self.reward_history[-self.context_length:]
        
        # Prepare inputs
        seq_len = len(self.state_history)
        
        # Pad with zeros if sequence is too short
        if seq_len < self.context_length:
            padding_len = self.context_length - seq_len
            padded_states = [np.zeros(6) for _ in range(padding_len)] + self.state_history  # Updated for 6D state space
            padded_actions = [np.zeros(1) for _ in range(padding_len)] + self.action_history
            padded_rewards = [0.0 for _ in range(padding_len)] + self.reward_history
        else:
            padded_states = self.state_history
            padded_actions = self.action_history
            padded_rewards = self.reward_history
        
        # Convert to tensors
        states = torch.tensor(padded_states, dtype=torch.float32)
        actions = torch.tensor(padded_actions, dtype=torch.float32).squeeze(-1)
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        if actions.shape[0] < self.context_length:
            # Pad actions if needed
            padding = torch.zeros(self.context_length - actions.shape[0])
            actions = torch.cat([padding, actions])
        actions = actions.unsqueeze(-1)  # Add action dimension back
        
        # Calculate returns-to-go (simple: target_return for all timesteps)
        returns_to_go = torch.full((self.context_length, 1), target_return, dtype=torch.float32)
        
        # Timesteps
        timesteps = torch.arange(self.context_length, dtype=torch.long)
        
        # Get action from model
        with torch.no_grad():
            action_pred = self.model.get_action(states, actions, returns_to_go, timesteps)
            
        # Convert from tanh output to servo range
        if self.model.action_tanh:
            action = action_pred.item() * self.action_scale + self.action_offset
        else:
            action = action_pred.item()
            
        # Clip to valid range
        action = np.clip(action, 0.0, 180.0)
        
        # Add normalized action to history (for next prediction)
        normalized_action = (action - self.action_offset) / self.action_scale
        self.action_history.append(np.array([normalized_action]))
        self.timestep += 1
        
        return action

# Example usage
if __name__ == "__main__":
    # Test model creation
    model = DecisionTransformer(
        state_dim=6,  # Updated for 6D state space
        action_dim=2,
        hidden_size=128,
        max_length=20
    )
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    states = torch.randn(batch_size, seq_len, 6)  # Updated for 6D state space
    actions = torch.randn(batch_size, seq_len, 1)
    returns_to_go = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    action_preds = model(states, actions, returns_to_go, timesteps)
    print(f"Model output shape: {action_preds.shape}")
    print("Decision Transformer test passed!")