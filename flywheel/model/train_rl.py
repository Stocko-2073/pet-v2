#!/usr/bin/env python3
"""
PPO training script for LSTM-based flywheel control.
"""

import torch
import numpy as np
import argparse
import time
import os
import json
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

from flywheel_env import FlywheelEnv
from lstm_rl_model import LSTMActorCritic, PPOBuffer, PPOTrainer

class RLTrainingManager:
    """Manager for PPO training on flywheel environment"""
    
    def __init__(
        self,
        env: FlywheelEnv,
        model: LSTMActorCritic,
        trainer: PPOTrainer,
        buffer_size: int = 10000,
        device: str = 'cpu'
    ):
        self.env = env
        self.model = model
        self.trainer = trainer
        self.device = device
        
        # Training buffer
        self.buffer = PPOBuffer(
            max_size=buffer_size,
            state_dim=model.state_dim,
            action_dim=model.action_dim,
            device=device
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
        # LSTM hidden state for episode continuity
        self.hidden_state = None
    
    def collect_experience(
        self,
        n_steps: int,
        max_episode_length: int = 200,
        render_frequency: int = 50
    ) -> Dict[str, Any]:
        """Collect experience using current policy"""
        
        self.model.eval()
        
        step_count = 0
        episode_count = 0
        total_reward = 0
        episode_reward = 0
        episode_length = 0
        
        # Reset environment and hidden state
        state = self.env.reset()
        self.hidden_state = None
        
        print(f"Collecting {n_steps} steps of experience...")
        start_time = time.time()
        
        while step_count < n_steps:
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            # Get action and value from model
            with torch.no_grad():
                action, log_prob, value, self.hidden_state = self.model.get_action_and_value(
                    state_tensor, self.hidden_state
                )

            # Take step in environment
            next_state, reward, done, info = self.env.step(action.squeeze())
            
            # Store experience in buffer
            self.buffer.store(
                state=state_tensor.squeeze(),
                action=action.squeeze(),
                log_prob=log_prob.squeeze(),
                value=value.squeeze(),
                reward=reward,
                done=done
            )
            
            # Update tracking variables
            step_count += 1
            episode_reward += reward
            episode_length += 1
            total_reward += reward
            state = next_state
            
            # Check for episode termination
            if done or episode_length >= max_episode_length:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_count += 1
                
                if episode_count % render_frequency == 0:
                    print(f"Episode {episode_count}: reward={episode_reward:.3f}, length={episode_length}")
                
                # Reset for next episode
                state = self.env.reset()
                self.hidden_state = None
                episode_reward = 0
                episode_length = 0
        
        collection_time = time.time() - start_time
        
        stats = {
            'steps_collected': step_count,
            'episodes_completed': episode_count,
            'total_reward': total_reward,
            'avg_episode_reward': np.mean(self.episode_rewards[-episode_count:]) if episode_count > 0 else 0,
            'avg_episode_length': np.mean(self.episode_lengths[-episode_count:]) if episode_count > 0 else 0,
            'collection_time': collection_time,
            'steps_per_second': step_count / collection_time
        }
        
        return stats
    
    def update_policy(self, n_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """Update policy using collected experience"""

        self.env.servo_enabled = False
        self.model.train()
        
        print(f"Updating policy for {n_epochs} epochs...")
        update_stats = self.trainer.update(self.buffer, n_epochs, batch_size)
        
        # Track losses
        if update_stats:
            self.training_losses.append(update_stats.get('total_loss', 0))
            self.policy_losses.append(update_stats.get('policy_loss', 0))
            self.value_losses.append(update_stats.get('value_loss', 0))
            self.entropy_losses.append(update_stats.get('entropy_loss', 0))
        
        # Clear buffer after update
        self.buffer.clear()

        self.env.servo_enabled = True
        
        return update_stats
    
    def train(
        self,
        total_timesteps: int,
        steps_per_update: int = 2048,
        max_episode_length: int = 200,
        update_epochs: int = 4,
        batch_size: int = 64,
        save_frequency: int = 10,
        log_frequency: int = 1,
        save_path: str = 'flywheel_ppo_model.pt'
    ):
        """Main training loop"""
        
        print(f"Starting PPO training for {total_timesteps} timesteps")
        print(f"Steps per update: {steps_per_update}")
        print(f"Max episode length: {max_episode_length}")
        print(f"Update epochs: {update_epochs}")
        print(f"Batch size: {batch_size}")
        
        timesteps_collected = 0
        update_count = 0
        
        # Training metrics for logging
        all_update_stats = []
        
        while timesteps_collected < total_timesteps:
            # Collect experience
            collection_stats = self.collect_experience(
                n_steps=steps_per_update,
                max_episode_length=max_episode_length
            )
            
            timesteps_collected += collection_stats['steps_collected']
            update_count += 1
            
            # Update policy
            update_stats = self.update_policy(update_epochs, batch_size)
            all_update_stats.append(update_stats)
            
            # Logging
            if update_count % log_frequency == 0:
                recent_rewards = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                
                print(f"\nUpdate {update_count}")
                print(f"Timesteps: {timesteps_collected}/{total_timesteps}")
                print(f"Avg reward (last 20 episodes): {avg_reward:.3f}")
                print(f"Episodes completed: {len(self.episode_rewards)}")
                print(f"Collection stats: {collection_stats}")
                print(f"Update stats: {update_stats}")
                print("-" * 50)
            
            # Save model
            if update_count % save_frequency == 0:
                self.save_model(save_path, update_count, timesteps_collected)
        
        # Final save
        self.save_model(save_path, update_count, timesteps_collected)
        
        # Plot training curves
        self.plot_training_curves()
        
        print(f"Training completed! Model saved to {save_path}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses,
            'update_stats': all_update_stats
        }
    
    def save_model(self, save_path: str, update_count: int, timesteps: int):
        """Save model and training state"""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'config': {
                'state_dim': self.model.state_dim,
                'action_dim': self.model.action_dim,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers
            },
            'training_stats': {
                'update_count': update_count,
                'timesteps': timesteps,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'training_losses': self.training_losses,
                'policy_losses': self.policy_losses,
                'value_losses': self.value_losses,
                'entropy_losses': self.entropy_losses
            }
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def plot_training_curves(self, save_path: str = 'rl_training_curves.png'):
        """Plot training progress"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            
            # Moving average
            if len(self.episode_rewards) > 10:
                window = min(50, len(self.episode_rewards) // 4)
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', alpha=0.7, label=f'MA({window})')
                axes[0, 0].legend()
        
        # Episode lengths
        if self.episode_lengths:
            axes[0, 1].plot(self.episode_lengths)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
        
        # Training losses
        if self.training_losses:
            axes[1, 0].plot(self.training_losses, label='Total Loss')
            if self.policy_losses:
                axes[1, 0].plot(self.policy_losses, label='Policy Loss')
            if self.value_losses:
                axes[1, 0].plot(self.value_losses, label='Value Loss')
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
        
        # Entropy loss
        if self.entropy_losses:
            axes[1, 1].plot(self.entropy_losses)
            axes[1, 1].set_title('Entropy Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Entropy Loss')
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")

def train_ppo_flywheel(
    port: str,
    total_timesteps: int = 50000,
    steps_per_update: int = 2048,
    hidden_size: int = 128,
    num_layers: int = 2,
    learning_rate: float = 3e-4,
    max_episode_length: int = 200,
        steps_per_action: int = 1,
        save_path: str = 'flywheel_ppo_model.pt',
    device: str = 'cpu'
):
    """Train PPO on flywheel control task"""
    
    print("Setting up flywheel environment...")
    env = FlywheelEnv(port,
                      max_episode_steps=max_episode_length,
                      position_history_size=2000,
                      steps_per_action=steps_per_action)

    try:
        # Create model
        print("Creating LSTM Actor-Critic model...")
        model = LSTMActorCritic(
            state_dim=6,  # [time, pwm, current, voltage, position, servo_enabled]
            action_dim=2,  # [servo_position, servo_enable]
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # Create trainer
        trainer = PPOTrainer(model, learning_rate=learning_rate, device=device)
        
        # Create training manager
        training_manager = RLTrainingManager(env, model, trainer, device=device)
        
        # Train
        results = training_manager.train(
            total_timesteps=total_timesteps,
            steps_per_update=steps_per_update,
            max_episode_length=max_episode_length,
            save_path=save_path
        )
        
        return results
        
    finally:
        env.close()

def test_trained_ppo_model(
    port: str, 
    model_path: str = 'flywheel_ppo_model.pt', 
    num_episodes: int = 5,
        max_episode_length: int = 200,
        steps_per_action: int = 1
):
    """Test trained PPO model"""
    
    print(f"Testing PPO model from {model_path}")
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        model = LSTMActorCritic(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded successfully. Config: {config}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test with environment
    env = FlywheelEnv(port, max_episode_steps=max_episode_length, steps_per_action=steps_per_action)

    try:
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            hidden_state = None
            episode_reward = 0
            episode_length = 0
            
            print(f"\nEpisode {episode + 1}:")
            
            for step in range(max_episode_length):
                # Get action from model
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    action, _, _, hidden_state = model.get_action_and_value(
                        state_tensor, hidden_state, deterministic=True
                    )
                
                # Take step
                next_state, reward, done, info = env.step(action.squeeze())

                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if step % 20 == 0:
                    print(f"  Step {step}: action={servo_action:.1f}, reward={reward:.3f}")
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"  Total reward: {episode_reward:.3f}, Length: {episode_length}")
        
        print(f"\nTest Results:")
        print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO for flywheel control')
    parser.add_argument('port', help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode: train or test')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total training timesteps')
    parser.add_argument('--steps-per-update', type=int, default=2048, help='Steps per policy update')
    parser.add_argument('--hidden-size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max-episode-length', type=int, default=400, help='Maximum episode length')
    parser.add_argument('--model-path', default='flywheel_ppo_model.pt', help='Model save/load path')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--steps-per-action', type=int, default=1, help='Steps per action')
    parser.add_argument('--test-episodes', type=int, default=5, help='Number of test episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ppo_flywheel(
            port=args.port,
            total_timesteps=args.timesteps,
            steps_per_update=args.steps_per_update,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            max_episode_length=args.max_episode_length,
            steps_per_action=args.steps_per_action,
            save_path=args.model_path,
            device=args.device
        )
    else:  # test
        test_trained_ppo_model(
            port=args.port,
            model_path=args.model_path,
            num_episodes=args.test_episodes,
            max_episode_length=args.max_episode_length,
            steps_per_action=args.steps_per_action
        )