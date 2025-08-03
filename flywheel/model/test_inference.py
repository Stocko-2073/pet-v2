#!/usr/bin/env python3
"""
Standalone inference test for trained models (Decision Transformer and LSTM).
"""

import torch
import numpy as np
import argparse
import time
from decision_transformer import FlywheelController
from rl_controller import LSTMFlywheelController
from flywheel_env import FlywheelEnv

def test_dt_inference(port: str, model_path: str = 'flywheel_dt_model.pt', num_steps: int = 100, target_rtg: float = 50.0):
    """Test inference with the trained Decision Transformer model"""
    
    print(f"Loading model from {model_path}")
    
    # Load model config
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        training_info = checkpoint.get('training_info', {})
        
        print(f"Model config: {config}")
        print(f"Training info: {training_info}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create controller
    print(f"Creating controller with target RTG: {target_rtg}")
    controller = FlywheelController(
        model_path, 
        context_length=config['max_length']
    )
    
    # Test with environment
    print(f"Connecting to flywheel on port {port}")
    env = FlywheelEnv(port, max_episode_steps=num_steps)
    
    try:
        # Reset environment and controller
        state = env.reset()
        controller.reset()
        
        print(f"Initial state: {state}")
        print("\nStarting inference test...")
        print("Step | Action | Reward | State")
        print("-" * 40)
        
        total_reward = 0
        actions = []
        rewards = []
        states = [state.copy()]
        
        for step in range(num_steps):
            # Get action from trained model
            action = controller.get_action(state)
            actions.append(action)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            rewards.append(reward)
            states.append(next_state.copy())
            
            # Print progress
            if step % 10 == 0 or step < 10:
                print(f"{step:4d} | {action:6.1f} | {reward:6.3f} | {next_state}")
            
            state = next_state
            
            if done:
                print(f"Episode ended at step {step}")
                break
        
        print(f"\nTest completed!")
        print(f"Total steps: {step + 1}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Average reward: {total_reward / (step + 1):.3f}")
        
        # Print some statistics
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        print(f"\nAction statistics:")
        print(f"  Mean: {actions.mean():.1f}")
        print(f"  Std:  {actions.std():.1f}")
        print(f"  Min:  {actions.min():.1f}")
        print(f"  Max:  {actions.max():.1f}")
        
        print(f"\nReward statistics:")
        print(f"  Mean: {rewards.mean():.3f}")
        print(f"  Std:  {rewards.std():.3f}")
        print(f"  Min:  {rewards.min():.3f}")
        print(f"  Max:  {rewards.max():.3f}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
    finally:
        env.close()
        print("Environment closed")

def test_lstm_inference(port: str, model_path: str = 'flywheel_ppo_model.pt', num_steps: int = 100, deterministic: bool = True):
    """Test inference with the trained LSTM model"""
    
    print(f"Loading LSTM model from {model_path}")
    
    # Create LSTM controller
    try:
        controller = LSTMFlywheelController(model_path, deterministic=deterministic)
        print(f"LSTM controller created successfully")
        
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return
    
    # Test with environment
    print(f"Connecting to flywheel on port {port}")
    env = FlywheelEnv(port, max_episode_steps=num_steps)
    
    try:
        # Reset environment and controller
        state = env.reset()
        controller.reset()
        
        print(f"Initial state: {state}")
        print("\nStarting LSTM inference test...")
        print("Step | Action | Value  | Reward | State")
        print("-" * 50)
        
        total_reward = 0
        actions = []
        rewards = []
        values = []
        states = [state.copy()]
        
        for step in range(num_steps):
            # Get action and info from trained LSTM model
            info = controller.get_action_with_info(state)
            action = info['action']
            value = info['value']
            
            actions.append(action)
            values.append(value)
            
            # Take action in environment
            next_state, reward, done, env_info = env.step(action)
            
            total_reward += reward
            rewards.append(reward)
            states.append(next_state.copy())
            
            # Print progress
            if step % 10 == 0 or step < 10:
                print(f"{step:4d} | {action:6.1f} | {value:6.3f} | {reward:6.3f} | {next_state}")
            
            state = next_state
            
            if done:
                print(f"Episode ended at step {step}")
                break
        
        print(f"\nLSTM test completed!")
        print(f"Total steps: {step + 1}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Average reward: {total_reward / (step + 1):.3f}")
        
        # Print statistics
        actions = np.array(actions)
        rewards = np.array(rewards)
        values = np.array(values)
        
        print(f"\nAction statistics:")
        print(f"  Mean: {actions.mean():.1f}")
        print(f"  Std:  {actions.std():.1f}")
        print(f"  Min:  {actions.min():.1f}")
        print(f"  Max:  {actions.max():.1f}")
        
        print(f"\nValue statistics:")
        print(f"  Mean: {values.mean():.3f}")
        print(f"  Std:  {values.std():.3f}")
        print(f"  Min:  {values.min():.3f}")
        print(f"  Max:  {values.max():.3f}")
        
        print(f"\nReward statistics:")
        print(f"  Mean: {rewards.mean():.3f}")
        print(f"  Std:  {rewards.std():.3f}")
        print(f"  Min:  {rewards.min():.3f}")
        print(f"  Max:  {rewards.max():.3f}")
        
        # Controller performance stats
        perf_stats = controller.get_performance_stats()
        print(f"\nController performance:")
        for key, value in perf_stats.items():
            print(f"  {key}: {value:.4f}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
    finally:
        env.close()
        print("Environment closed")

def compare_models(port: str, dt_model: str = 'flywheel_dt_model.pt', lstm_model: str = 'flywheel_ppo_model.pt', num_steps: int = 100):
    """Compare Decision Transformer and LSTM models side by side"""
    
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    # Load both controllers
    try:
        print("Loading Decision Transformer...")
        dt_checkpoint = torch.load(dt_model, map_location='cpu', weights_only=False)
        dt_config = dt_checkpoint['config']
        dt_controller = FlywheelController(dt_model, context_length=dt_config['max_length'])
        print("✓ Decision Transformer loaded")
        
        print("Loading LSTM...")
        lstm_controller = LSTMFlywheelController(lstm_model, deterministic=True)
        print("✓ LSTM loaded")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Test environment
    env = FlywheelEnv(port, max_episode_steps=num_steps)
    
    try:
        results = {'dt': [], 'lstm': []}
        
        for model_name, controller in [('dt', dt_controller), ('lstm', lstm_controller)]:
            print(f"\n{'='*20} Testing {model_name.upper()} Model {'='*20}")
            
            state = env.reset()
            controller.reset()
            
            total_reward = 0
            actions = []
            
            for step in range(num_steps):
                if model_name == 'dt':
                    action = controller.get_action(state)
                else:  # lstm
                    info = controller.get_action_with_info(state)
                    action = info['action']
                
                actions.append(action)
                next_state, reward, done, info = env.step(action)
                
                total_reward += reward
                state = next_state
                
                if step % 20 == 0:
                    print(f"Step {step}: action={action:.1f}, reward={reward:.3f}")
                
                if done:
                    break
            
            results[model_name] = {
                'total_reward': total_reward,
                'avg_reward': total_reward / (step + 1),
                'actions': np.array(actions),
                'steps': step + 1
            }
        
        # Print comparison
        print(f"\n{'='*20} COMPARISON RESULTS {'='*20}")
        print(f"Decision Transformer:")
        print(f"  Total reward: {results['dt']['total_reward']:.3f}")
        print(f"  Avg reward:   {results['dt']['avg_reward']:.3f}")
        print(f"  Steps:        {results['dt']['steps']}")
        
        print(f"\nLSTM:")
        print(f"  Total reward: {results['lstm']['total_reward']:.3f}")
        print(f"  Avg reward:   {results['lstm']['avg_reward']:.3f}")
        print(f"  Steps:        {results['lstm']['steps']}")
        
        # Performance difference
        reward_diff = results['lstm']['total_reward'] - results['dt']['total_reward']
        print(f"\nPerformance difference (LSTM - DT): {reward_diff:.3f}")
        
        if reward_diff > 0:
            print("→ LSTM performed better")
        elif reward_diff < 0:
            print("→ Decision Transformer performed better")
        else:
            print("→ Both models performed equally")
        
        # Action analysis
        print(f"\nAction analysis:")
        dt_actions = results['dt']['actions']
        lstm_actions = results['lstm']['actions']
        
        print(f"DT action range:   [{dt_actions.min():.1f}, {dt_actions.max():.1f}] (std: {dt_actions.std():.2f})")
        print(f"LSTM action range: [{lstm_actions.min():.1f}, {lstm_actions.max():.1f}] (std: {lstm_actions.std():.2f})")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
    finally:
        env.close()

def test_model_loading(model_path: str = 'flywheel_dt_model.pt', model_type: str = 'auto'):
    """Test just the model loading without hardware"""
    print(f"Testing model loading from {model_path}")
    
    # Auto-detect model type from filename or checkpoint
    if model_type == 'auto':
        if 'ppo' in model_path.lower() or 'lstm' in model_path.lower():
            model_type = 'lstm'
        else:
            model_type = 'dt'
    
    print(f"Detected model type: {model_type}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✓ Model file loaded successfully")
        
        # Common required keys
        if 'config' not in checkpoint:
            print("✗ Missing 'config' in checkpoint")
            return False
        print("✓ Found 'config' in checkpoint")
        
        config = checkpoint['config']
        print(f"✓ Model config: {config}")
        
        if model_type == 'dt':
            # Decision Transformer specific
            if 'model_state_dict' not in checkpoint:
                print("✗ Missing 'model_state_dict' in checkpoint")
                return False
            print("✓ Found 'model_state_dict' in checkpoint")
            
            # Try creating controller
            controller = FlywheelController(model_path, context_length=config['max_length'])
            print("✓ FlywheelController created successfully")
            
            # Test with dummy state
            dummy_state = np.array([0.0, 100.0, 0.0, 12.0, 0.0])  # [time, pwm, current, voltage, position]
            action = controller.get_action(dummy_state)
            print(f"✓ Generated action for dummy state: {action:.1f}")
            
        elif model_type == 'lstm':
            # LSTM specific
            if 'model_state_dict' not in checkpoint:
                print("✗ Missing 'model_state_dict' in checkpoint")
                return False
            print("✓ Found 'model_state_dict' in checkpoint")
            
            # Try creating controller
            controller = LSTMFlywheelController(model_path, deterministic=True)
            print("✓ LSTMFlywheelController created successfully")
            
            # Test with dummy state
            dummy_state = np.array([0.0, 100.0, 0.0, 12.0, 0.0])  # [time, pwm, current, voltage, position]
            info = controller.get_action_with_info(dummy_state)
            print(f"✓ Generated action for dummy state: {info['action']:.1f}")
            print(f"✓ Value estimate: {info['value']:.3f}")
            print(f"✓ Inference time: {info['inference_time']*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test inference with trained models (DT and LSTM)')
    parser.add_argument('--port', help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--model-path', default='flywheel_dt_model.pt', help='Path to trained model')
    parser.add_argument('--model-type', choices=['dt', 'lstm', 'auto'], default='auto', 
                        help='Model type (auto-detect by default)')
    parser.add_argument('--steps', type=int, default=10000, help='Number of test steps')
    parser.add_argument('--target-rtg', type=float, default=50.0, help='Target return-to-go (DT only)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions (LSTM only)')
    parser.add_argument('--test-loading-only', action='store_true', help='Only test model loading (no hardware)')
    parser.add_argument('--compare', action='store_true', help='Compare DT and LSTM models')
    parser.add_argument('--dt-model', default='flywheel_dt_model.pt', help='Decision Transformer model path (for comparison)')
    parser.add_argument('--lstm-model', default='flywheel_ppo_model.pt', help='LSTM model path (for comparison)')
    
    args = parser.parse_args()
    
    if args.test_loading_only:
        test_model_loading(args.model_path, args.model_type)
    elif args.compare:
        if not args.port:
            print("Error: --port is required for hardware testing")
            exit(1)
        compare_models(args.port, args.dt_model, args.lstm_model, args.steps)
    else:
        if not args.port:
            print("Error: --port is required for hardware testing")
            print("Use --test-loading-only to test without hardware")
            exit(1)
        
        # Auto-detect model type if needed
        model_type = args.model_type
        if model_type == 'auto':
            if 'ppo' in args.model_path.lower() or 'lstm' in args.model_path.lower():
                model_type = 'lstm'
            else:
                model_type = 'dt'
        
        if model_type == 'dt':
            test_dt_inference(args.port, args.model_path, args.steps, args.target_rtg)
        elif model_type == 'lstm':
            test_lstm_inference(args.port, args.model_path, args.steps, args.deterministic)