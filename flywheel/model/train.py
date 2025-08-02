#!/usr/bin/env python3
"""
Training script for Decision Transformer on flywheel control task.
Collects data from the environment and trains the transformer model.
"""

import torch
import numpy as np
import argparse
import time
import os
import json
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt

from flywheel_env import FlywheelEnv
from decision_transformer import DecisionTransformer, DecisionTransformerTrainer, FlywheelController

@dataclass
class TrajectoryStep:
    """Single step in a trajectory"""
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool
    timestep: int

@dataclass
class Trajectory:
    """Complete trajectory/episode"""
    steps: List[TrajectoryStep]
    total_return: float
    length: int

class DataCollector:
    """Collect training data from the flywheel environment"""
    
    def __init__(self, env: FlywheelEnv, random_policy: bool = True):
        self.env = env
        self.random_policy = random_policy
        self.trajectories = []
        
    def collect_random_trajectory(self, max_steps: int = 200) -> Trajectory:
        """Collect one trajectory using random actions"""
        state = self.env.reset()
        steps = []
        total_return = 0.0
        
        for t in range(max_steps):
            # Random action
            action = np.random.uniform(self.env.action_space_low, self.env.action_space_high)
            
            next_state, reward, done, info = self.env.step(action)
            
            step = TrajectoryStep(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done,
                timestep=t
            )
            steps.append(step)
            
            total_return += reward
            state = next_state
            
            if done:
                break
                
        trajectory = Trajectory(
            steps=steps,
            total_return=total_return,
            length=len(steps)
        )
        
        return trajectory
    
    def collect_data(self, num_trajectories: int, max_steps_per_episode: int = 200) -> List[Trajectory]:
        """Collect multiple trajectories"""
        trajectories = []
        
        print(f"Collecting {num_trajectories} trajectories...")
        
        for i in range(num_trajectories):
            print(f"Collecting trajectory {i+1}/{num_trajectories}")
            
            try:
                traj = self.collect_random_trajectory(max_steps_per_episode)
                trajectories.append(traj)
                print(f"  Length: {traj.length}, Return: {traj.total_return:.3f}")
                
                # Brief pause between episodes
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error collecting trajectory: {e}")
                continue
        
        self.trajectories.extend(trajectories)
        return trajectories

class DecisionTransformerDataset:
    """Dataset for training Decision Transformer"""
    
    def __init__(
        self, 
        trajectories: List[Trajectory],
        context_length: int = 20,
        rtg_scale: float = 1000.0  # Scale return-to-go values
    ):
        self.trajectories = trajectories
        self.context_length = context_length
        self.rtg_scale = rtg_scale
        
        # Pre-process trajectories
        self.sequences = []
        self._create_sequences()
        
    def _create_sequences(self):
        """Create training sequences from trajectories"""
        for traj in self.trajectories:
            if len(traj.steps) < 2:
                continue
                
            # Calculate returns-to-go
            returns_to_go = []
            cumulative_return = 0.0
            for step in reversed(traj.steps):
                cumulative_return += step.reward
                returns_to_go.append(cumulative_return)
            returns_to_go.reverse()
            
            # Create sequences of length context_length
            for start_idx in range(len(traj.steps)):
                end_idx = min(start_idx + self.context_length, len(traj.steps))
                
                if end_idx - start_idx < 2:  # Need at least 2 steps
                    continue
                
                seq_states = []
                seq_actions = []
                seq_rtg = []
                seq_timesteps = []
                
                for i in range(start_idx, end_idx):
                    step = traj.steps[i]
                    seq_states.append(step.state)
                    
                    # Normalize action to [-1, +1] range for tanh model
                    normalized_action = (step.action - 90.0) / 90.0  # Convert 0-180 to -1,+1
                    seq_actions.append([normalized_action])
                    
                    seq_rtg.append([returns_to_go[i] / self.rtg_scale])  # Normalize RTG
                    seq_timesteps.append(step.timestep)
                
                # Target actions are the actions we want to predict (already normalized)
                target_actions = seq_actions.copy()
                
                self.sequences.append({
                    'states': np.array(seq_states),
                    'actions': np.array(seq_actions),
                    'returns_to_go': np.array(seq_rtg),
                    'timesteps': np.array(seq_timesteps),
                    'target_actions': np.array(target_actions)
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def get_batch(self, batch_size: int, device: str = 'cpu'):
        """Get a batch of sequences"""
        indices = np.random.choice(len(self.sequences), batch_size, replace=True)
        
        batch_states = []
        batch_actions = []
        batch_rtg = []
        batch_timesteps = []
        batch_target_actions = []
        
        max_len = 0
        for idx in indices:
            seq = self.sequences[idx]
            max_len = max(max_len, len(seq['states']))
        
        for idx in indices:
            seq = self.sequences[idx]
            seq_len = len(seq['states'])
            
            # Pad sequences to max_len
            if seq_len < max_len:
                pad_len = max_len - seq_len
                states = np.concatenate([np.zeros((pad_len, 5)), seq['states']], axis=0)
                actions = np.concatenate([np.zeros((pad_len, 1)), seq['actions']], axis=0)
                rtg = np.concatenate([np.zeros((pad_len, 1)), seq['returns_to_go']], axis=0)
                timesteps = np.concatenate([np.zeros(pad_len), seq['timesteps']], axis=0)
                target_actions = np.concatenate([np.zeros((pad_len, 1)), seq['target_actions']], axis=0)
            else:
                states = seq['states']
                actions = seq['actions']
                rtg = seq['returns_to_go']
                timesteps = seq['timesteps']
                target_actions = seq['target_actions']
            
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rtg.append(rtg)
            batch_timesteps.append(timesteps)
            batch_target_actions.append(target_actions)
        
        return {
            'states': torch.tensor(np.array(batch_states), dtype=torch.float32, device=device),
            'actions': torch.tensor(np.array(batch_actions), dtype=torch.float32, device=device),
            'returns_to_go': torch.tensor(np.array(batch_rtg), dtype=torch.float32, device=device),
            'timesteps': torch.tensor(np.array(batch_timesteps), dtype=torch.long, device=device),
            'target_actions': torch.tensor(np.array(batch_target_actions), dtype=torch.float32, device=device)
        }

def train_decision_transformer(
    port: str,
    num_trajectories: int = 50,
    context_length: int = 20,
    hidden_size: int = 128,
    n_layers: int = 3,
    n_heads: int = 4,
    learning_rate: float = 1e-5,  # Reduced from 1e-4
    batch_size: int = 64,
    num_epochs: int = 100,
    save_path: str = 'flywheel_dt_model.pt',
    device: str = 'cpu'
):
    """Train Decision Transformer on flywheel data"""
    
    print("Setting up flywheel environment...")
    env = FlywheelEnv(port, max_episode_steps=200)
    
    try:
        # Collect data
        print("Collecting training data...")
        collector = DataCollector(env)
        trajectories = collector.collect_data(num_trajectories)
        
        print(f"Collected {len(trajectories)} trajectories")
        if len(trajectories) == 0:
            print("No trajectories collected! Check your hardware connection.")
            return
        
        # Create dataset
        print("Creating dataset...")
        dataset = DecisionTransformerDataset(trajectories, context_length)
        print(f"Dataset contains {len(dataset)} sequences")
        
        if len(dataset) == 0:
            print("No training sequences created!")
            return
        
        # Create model
        print("Creating model...")
        model = DecisionTransformer(
            state_dim=5,
            action_dim=1,
            hidden_size=hidden_size,
            max_length=context_length,
            n_layer=n_layers,
            n_head=n_heads
        )
        
        # Create trainer
        trainer = DecisionTransformerTrainer(model, learning_rate, device=device)
        
        # Training loop
        print(f"Starting training for {num_epochs} epochs...")
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            num_batches = max(1, len(dataset) // batch_size)
            
            for batch_idx in range(num_batches):
                batch = dataset.get_batch(batch_size, device)
                
                loss = trainer.train_step(
                    batch['states'],
                    batch['actions'], 
                    batch['returns_to_go'],
                    batch['timesteps'],
                    batch['target_actions']
                )
                
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        # Save model
        print(f"Saving model to {save_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'state_dim': 5,
                'action_dim': 1,
                'hidden_size': hidden_size,
                'max_length': context_length,
                'n_layer': n_layers,
                'n_head': n_heads
            },
            'training_info': {
                'num_trajectories': len(trajectories),
                'num_sequences': len(dataset),
                'final_loss': losses[-1] if losses else 0.0
            }
        }, save_path)
        
        # Plot training curve
        if len(losses) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig('training_curve.png')
            print("Training curve saved to training_curve.png")
        
        print("Training completed!")
        
    finally:
        env.close()

def test_trained_model(port: str, model_path: str = 'flywheel_dt_model.pt', num_steps: int = 100):
    """Test the trained model"""
    print(f"Testing model from {model_path}")
    
    # Load model config
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create controller
    controller = FlywheelController(model_path, context_length=config['max_length'])
    
    # Test with environment
    env = FlywheelEnv(port, max_episode_steps=num_steps)
    
    try:
        state = env.reset()
        controller.reset()
        
        total_reward = 0
        for step in range(num_steps):
            action = controller.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            state = next_state
            
            if step % 10 == 0:
                print(f"Step {step}: action={action:.1f}, reward={reward:.3f}")
            
            if done:
                break
        
        print(f"Test completed! Total reward: {total_reward:.3f}")
        
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Decision Transformer for flywheel control')
    parser.add_argument('port', help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode: train or test')
    parser.add_argument('--trajectories', type=int, default=20, help='Number of trajectories to collect')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--context-length', type=int, default=20, help='Context length')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size')
    parser.add_argument('--model-path', default='flywheel_dt_model.pt', help='Model save/load path')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_decision_transformer(
            port=args.port,
            num_trajectories=args.trajectories,
            context_length=args.context_length,
            hidden_size=args.hidden_size,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.model_path,
            device=args.device
        )
    else:  # test
        test_trained_model(args.port, args.model_path)