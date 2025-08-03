#!/usr/bin/env python3
"""
Real-time LSTM controller for flywheel control using trained PPO model.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time

from lstm_rl_model import LSTMActorCritic

class LSTMFlywheelController:
    """
    Real-time controller using trained LSTM Actor-Critic model.
    
    Maintains LSTM hidden state across action predictions for continuous control.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        action_scale: float = 90.0,  # Scale tanh output to 0-180 degrees
        action_offset: float = 90.0,
        deterministic: bool = True
    ):
        self.device = device
        self.action_scale = action_scale
        self.action_offset = action_offset
        self.deterministic = deterministic
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(device)
        
        # LSTM hidden state (maintained across predictions)
        self.hidden_state = None
        
        # State normalization parameters (should match training)
        self.state_mean = None
        self.state_std = None
        
        # Performance tracking
        self.inference_times = []
        self.action_history = []
        
    def _load_model(self, model_path: str) -> LSTMActorCritic:
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'config' not in checkpoint:
                raise ValueError("Model checkpoint missing 'config' key")
            
            config = checkpoint['config']
            
            # Create model with saved configuration
            model = LSTMActorCritic(
                state_dim=config['state_dim'],
                action_dim=config['action_dim'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers']
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load training stats if available for state normalization
            if 'training_stats' in checkpoint:
                training_stats = checkpoint['training_stats']
                print(f"Model trained for {training_stats.get('timesteps', 'unknown')} timesteps")
                print(f"Final episode reward: {training_stats.get('episode_rewards', [0])[-1] if training_stats.get('episode_rewards') else 'unknown'}")
            
            print(f"Loaded LSTM model: {config}")
            return model
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise
    
    def reset(self):
        """Reset controller state for new episode"""
        self.hidden_state = None
        self.action_history.clear()
        self.inference_times.clear()
        print("Controller reset")
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state for model input.
        
        Note: This should match the normalization used during training.
        For now, we use a simple normalization assuming states are roughly in expected ranges.
        """
        # Define expected state ranges (from flywheel_env.py)
        state_min = np.array([0, 0, -5000, 0, -100000], dtype=np.float32)
        state_max = np.array([1e9, 1023, 5000, 15, 100000], dtype=np.float32)
        
        # Normalize to [-1, 1] range
        normalized = 2.0 * (state - state_min) / (state_max - state_min) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def get_action(
        self,
        state: np.ndarray,
        timing: bool = False
    ) -> float:
        """
        Get action for current state.
        
        Args:
            state: Current state (5-dim numpy array)
            timing: Whether to track inference timing
            
        Returns:
            Servo position (0-180 degrees)
        """
        start_time = time.time() if timing else None
        
        # Normalize state
        normalized_state = self.normalize_state(state)
        
        # Convert to tensor with batch and sequence dimensions
        state_tensor = torch.tensor(
            normalized_state, 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0).unsqueeze(0)  # (batch=1, seq_len=1, state_dim)
        
        # Get action from model
        with torch.no_grad():
            action, log_prob, value, self.hidden_state = self.model.get_action_and_value(
                state_tensor, 
                self.hidden_state, 
                deterministic=self.deterministic
            )
        
        # Convert action from [-1, 1] to [0, 180] degrees
        action_np = action.squeeze().cpu().numpy()
        if isinstance(action_np, np.ndarray) and action_np.shape == ():
            action_np = action_np.item()
        
        servo_action = action_np * self.action_scale + self.action_offset
        servo_action = np.clip(servo_action, 0.0, 180.0)
        
        # Track action history
        self.action_history.append(servo_action)
        
        # Track timing if requested
        if timing and start_time is not None:
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
        
        return servo_action
    
    def get_action_with_info(
        self,
        state: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get action with additional information for debugging.
        
        Returns:
            Dictionary with action, value, log_prob, and other info
        """
        start_time = time.time()
        
        # Normalize state
        normalized_state = self.normalize_state(state)
        
        # Convert to tensor
        state_tensor = torch.tensor(
            normalized_state, 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0).unsqueeze(0)
        
        # Get action and additional info
        with torch.no_grad():
            action, log_prob, value, self.hidden_state = self.model.get_action_and_value(
                state_tensor, 
                self.hidden_state, 
                deterministic=self.deterministic
            )
        
        # Convert action
        action_np = action.squeeze().cpu().numpy()
        if isinstance(action_np, np.ndarray) and action_np.shape == ():
            action_np = action_np.item()
        
        servo_action = action_np * self.action_scale + self.action_offset
        servo_action = np.clip(servo_action, 0.0, 180.0)
        
        inference_time = time.time() - start_time
        
        return {
            'action': servo_action,
            'raw_action': action_np,
            'value': value.squeeze().cpu().numpy().item(),
            'log_prob': log_prob.squeeze().cpu().numpy().item(),
            'normalized_state': normalized_state,
            'inference_time': inference_time,
            'hidden_state_norm': torch.norm(self.hidden_state[0]).item() if self.hidden_state else 0.0
        }
    
    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for current state"""
        normalized_state = self.normalize_state(state)
        state_tensor = torch.tensor(
            normalized_state, 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            _, values, _ = self.model(state_tensor, self.hidden_state)
            
        return values.squeeze().cpu().numpy().item()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        stats = {}
        
        if self.inference_times:
            stats.update({
                'avg_inference_time': np.mean(self.inference_times),
                'max_inference_time': np.max(self.inference_times),
                'min_inference_time': np.min(self.inference_times),
                'inference_hz': 1.0 / np.mean(self.inference_times),
                'total_inferences': len(self.inference_times)
            })
        
        if self.action_history:
            actions = np.array(self.action_history)
            stats.update({
                'action_mean': np.mean(actions),
                'action_std': np.std(actions),
                'action_min': np.min(actions),
                'action_max': np.max(actions),
                'action_range': np.max(actions) - np.min(actions)
            })
            
            # Action smoothness (variance of action differences)
            if len(actions) > 1:
                action_diffs = np.diff(actions)
                stats['action_smoothness'] = np.var(action_diffs)
        
        return stats
    
    def set_deterministic(self, deterministic: bool):
        """Set whether to use deterministic (mean) actions"""
        self.deterministic = deterministic
        print(f"Controller set to {'deterministic' if deterministic else 'stochastic'} mode")

# Utility functions for testing and evaluation

def compare_controllers(
    lstm_controller: LSTMFlywheelController,
    dt_controller,  # Decision Transformer controller
    env,
    num_episodes: int = 5,
    max_episode_length: int = 200
) -> Dict[str, Any]:
    """
    Compare LSTM controller with Decision Transformer controller.
    
    Returns comparison statistics.
    """
    results = {
        'lstm_rewards': [],
        'dt_rewards': [],
        'lstm_lengths': [],
        'dt_lengths': [],
        'lstm_performance': {},
        'dt_performance': {}
    }
    
    print(f"Comparing controllers over {num_episodes} episodes...")
    
    # Test LSTM controller
    print("Testing LSTM controller...")
    for episode in range(num_episodes):
        state = env.reset()
        lstm_controller.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_episode_length):
            action = lstm_controller.get_action(state, timing=True)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        results['lstm_rewards'].append(episode_reward)
        results['lstm_lengths'].append(episode_length)
    
    results['lstm_performance'] = lstm_controller.get_performance_stats()
    
    # Test Decision Transformer controller
    print("Testing Decision Transformer controller...")
    for episode in range(num_episodes):
        state = env.reset()
        dt_controller.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_episode_length):
            action = dt_controller.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        results['dt_rewards'].append(episode_reward)
        results['dt_lengths'].append(episode_length)
    
    # Compute comparison statistics
    results['comparison'] = {
        'lstm_avg_reward': np.mean(results['lstm_rewards']),
        'dt_avg_reward': np.mean(results['dt_rewards']),
        'lstm_avg_length': np.mean(results['lstm_lengths']),
        'dt_avg_length': np.mean(results['dt_lengths']),
        'reward_improvement': np.mean(results['lstm_rewards']) - np.mean(results['dt_rewards']),
        'length_difference': np.mean(results['lstm_lengths']) - np.mean(results['dt_lengths'])
    }
    
    return results

# Example usage and testing
if __name__ == "__main__":
    import argparse
    from flywheel_env import FlywheelEnv
    
    parser = argparse.ArgumentParser(description='Test LSTM controller')
    parser.add_argument('--model', required=True, help='Path to trained LSTM model')
    parser.add_argument('--port', help='Serial port for hardware testing')
    parser.add_argument('--episodes', type=int, default=3, help='Number of test episodes')
    parser.add_argument('--max-length', type=int, default=200, help='Max episode length')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions')
    parser.add_argument('--test-model-only', action='store_true', help='Test model loading only')
    
    args = parser.parse_args()
    
    # Test model loading
    print(f"Loading LSTM controller from {args.model}")
    try:
        controller = LSTMFlywheelController(
            args.model, 
            deterministic=args.deterministic
        )
        print("✓ Controller loaded successfully")
    except Exception as e:
        print(f"✗ Error loading controller: {e}")
        exit(1)
    
    if args.test_model_only:
        # Test with dummy state
        dummy_state = np.array([0.0, 100.0, 0.0, 12.0, 0.0])  # [time, pwm, current, voltage, position]
        
        print("Testing with dummy state...")
        for i in range(5):
            info = controller.get_action_with_info(dummy_state)
            print(f"Step {i}: action={info['action']:.1f}, value={info['value']:.3f}, time={info['inference_time']*1000:.1f}ms")
        
        stats = controller.get_performance_stats()
        print(f"Performance stats: {stats}")
        
    else:
        if not args.port:
            print("Error: --port required for hardware testing")
            exit(1)
        
        # Test with hardware
        print(f"Testing with hardware on port {args.port}")
        env = FlywheelEnv(args.port, max_episode_steps=args.max_length)
        
        try:
            episode_rewards = []
            
            for episode in range(args.episodes):
                state = env.reset()
                controller.reset()
                episode_reward = 0
                
                print(f"\nEpisode {episode + 1}:")
                
                for step in range(args.max_length):
                    info = controller.get_action_with_info(state)
                    next_state, reward, done, env_info = env.step(info['action'])
                    
                    episode_reward += reward
                    state = next_state
                    
                    if step % 20 == 0:
                        print(f"  Step {step}: action={info['action']:.1f}, value={info['value']:.3f}, reward={reward:.3f}")
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                print(f"  Episode reward: {episode_reward:.3f}")
            
            print(f"\nTest Results:")
            print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
            
            # Performance stats
            stats = controller.get_performance_stats()
            print(f"Performance stats: {stats}")
            
        finally:
            env.close()