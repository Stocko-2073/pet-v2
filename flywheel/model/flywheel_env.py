#!/usr/bin/env python3
"""
Gym-style environment wrapper for the flywheel control system.
Provides standardized RL interface for transformer-based control.
"""

import numpy as np
import time
from typing import Optional, Tuple, Dict, Any, List
import sys
import os
from collections import deque
import random
import threading

# Add parent directory to path to import host_comm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from host_comm import FlywheelComm,SensorData,ERROR_MESSAGES


class FlywheelEnv:
    """
    Reinforcement Learning environment for flywheel control.
    
    State: [time_us, pwm_value, current_mA, voltage_V, position]
    Action: servo_position (0-180 degrees)
    """

    def _communication_loop(self):
        """Thread function: Unified serial communication - handles both servo control and data collection"""
        while True:
            if self.comm:
                # Send servo command and track statistics
                try:
                    if self.reset_env:
                        self.reset_env = False
                        # Reset encoder position
                        self.comm.reset_encoder()
                        time.sleep(0.1)  # Allow reset to complete

                        # Set servo to neutral position
                        self.comm.set_servo_position(90.0)
                        time.sleep(0.1)

                    # Update servo position
                    if self.servo_position != self.last_servo_position:
                        if self.comm.set_servo_position(self.servo_position):
                            self.last_servo_position = self.servo_position
                    # Read sensor data
                    data = self.comm.read_sensor_data()
                    if data:
                        self.sensor_data = data

                    # Check for errors
                    error = self.comm.read_error()
                    if error is not None:
                        error_msg = ERROR_MESSAGES.get(error, f"Unknown error: {error}")
                        print(f"Protocol error: {error_msg}")

                except Exception:
                    # Suppress timeout exception messages to avoid spam
                    pass

            # Small sleep to prevent overwhelming the system
            time.sleep(0.001)  # 1ms

    def __init__(self, port: str, baudrate: int = 2000000, max_episode_steps: int = 1000,
                 position_history_size: int = 20, target_tolerance: float = 100.0):
        self.port = port
        self.baudrate = baudrate
        self.max_episode_steps = max_episode_steps
        self.comm = None

        self.servo_position = 0
        self.last_servo_position = 180
        self.sensor_data = None
        self.reset_env = False
        self.communication_thread: Optional[threading.Thread] = None
        self.communication_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.communication_thread.start()


        # Home the servo and calibrate the position sensor
        if not self.comm:
            self.connect()
        print("Homing Servo")
        self.servo_position=0
        time.sleep(3)
        while self.sensor_data is None:
            time.sleep(0.01)
        last_sensor_data = self.sensor_data
        while True:
            time.sleep(0.01)
            if self.sensor_data.position == last_sensor_data.position:
                break
        self.reset_env = True
        while self.reset_env:
            time.sleep(0.01)
        print("done.")


        # State space: 5 dimensions
        self.observation_space_low = np.array([0, 0, -5000, 0, -100000], dtype=np.float32)
        self.observation_space_high = np.array([1e9, 1023, 5000, 15, 100000], dtype=np.float32)
        
        # Action space: servo position 0-180 degrees
        self.action_space_low = 0.0
        self.action_space_high = 180.0
        
        # Episode tracking
        self.current_step = 0
        self.episode_start_time = None
        self.last_state = None
        
        # Target position management
        self.target_position = 0
        self.target_tolerance = target_tolerance
        self.target_change_probability = 0.1  # Chance to change target each episode
        self.possible_targets = [0, 45, 90, 135, 180]  # Example target positions
        
        # Position history for oscillation detection
        self.position_history_size = position_history_size
        self.position_history = deque(maxlen=position_history_size)
        self.position_velocity_history = deque(maxlen=position_history_size-1)
        self.last_position = None
        self.last_time = None
        
        # Stability tracking
        self.time_in_tolerance = 0
        self.consecutive_stable_steps = 0
        self.stability_threshold = 50.0  # Position velocity threshold for "stable"
        self.last_action = None  # For action smoothness reward

        self.reset()
        
    def connect(self):
        """Connect to the flywheel device"""
        if self.comm is None:
            self.comm = FlywheelComm(self.port, self.baudrate)
        return self.comm
    
    def disconnect(self):
        """Disconnect from the flywheel device"""
        if self.comm:
            self.comm.close()
            self.comm = None
    
    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode"""
        if not self.comm:
            self.connect()

        # Reset episode tracking
        self.current_step = 0
        self.episode_start_time = time.time()
        
        # Reset position history and tracking
        self.position_history.clear()
        self.position_velocity_history.clear()
        self.last_position = None
        self.last_time = None
        self.time_in_tolerance = 0
        self.consecutive_stable_steps = 0
        self.last_action = None

        self.servo_position=random.uniform(self.action_space_low,self.action_space_high)
        print(f"Moving servo position to {self.servo_position}")
        time.sleep(0.4)
        print(f"Servo moved to {self.servo_position}")

        # Randomly set a new target position for this episode
        if random.random() < self.target_change_probability or self.target_position == 0:
            self.target_position = random.choice(self.possible_targets)
        
        # Get initial state
        state = self._get_state()
        self.last_state = state
        
        return state
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Servo position (0-180 degrees)
            
        Returns:
            observation: Next state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        success = True

        # Clip action to valid range
        action = np.clip(action, self.action_space_low, self.action_space_high)
        
        # Send the action to device
        self.servo_position = action

        # Wait a bit for the action to take effect ~100Hz
        time.sleep(0.01)

        # Get new state
        new_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(self.last_state, action, new_state)
        
        # Check if episode is done
        self.current_step += 1
        done = (self.current_step >= self.max_episode_steps) or self._is_done(new_state)
        
        # Update last state
        self.last_state = new_state
        
        info = {
            'step': self.current_step,
            'action_success': success,
            'servo_position': action
        }
        
        return new_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state from sensors"""
        # Try to get sensor data with timeout
        sensor_data = self.sensor_data
        while sensor_data is None:
            sensor_data = self.sensor_data
            time.sleep(0.001)

        # Convert to normalized state vector
        raw_state = np.array([
            0,#float(sensor_data.time_us),
            float(sensor_data.pwm_value),
            sensor_data.current_mA,
            sensor_data.voltage_V,
            float(sensor_data.position)
        ], dtype=np.float32)
        
        # Update position history for oscillation detection
        self._update_position_history(raw_state[4], sensor_data.timestamp)
        
        # Normalize state to roughly [-1, 1] range for better learning
        normalized_state = self._normalize_state(raw_state)
        
        return normalized_state
    
    def _update_position_history(self, position: float, timestamp: float):
        """Update position history and calculate derivatives"""
        self.position_history.append(position)
        
        if self.last_position is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt > 0:
                velocity = (position - self.last_position) / dt
                self.position_velocity_history.append(velocity)
        
        self.last_position = position
        self.last_time = timestamp
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state values for better learning"""
        # Simple min-max normalization to [-1, 1]
        normalized = 2.0 * (state - self.observation_space_low) / (self.observation_space_high - self.observation_space_low) - 1.0
        return np.clip(normalized, -1.0, 1.0)
    
    def _denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """Convert normalized state back to original scale"""
        return self.observation_space_low + (normalized_state + 1.0) * (self.observation_space_high - self.observation_space_low) / 2.0
    
    def _calculate_reward(self, last_state: Optional[np.ndarray], action: float, new_state: np.ndarray) -> float:
        """
        Enhanced reward function for position control with oscillation penalty.
        Rewards: position accuracy, stability, low oscillation, energy efficiency
        """
        if last_state is None or len(self.position_history) < 2:
            return 0.0
        
        # Denormalize states for reward calculation
        new_real = self._denormalize_state(new_state)
        current_position = new_real[4]  # position is index 4
        current_mA = new_real[2]        # current is index 2
        
        # 1. Position Error Reward (exponential decay)
        position_error = abs(current_position - self.target_position)
        position_reward = np.exp(-position_error / 200.0)  # Max reward 1.0 at perfect position
        
        # 2. Stability Reward (low velocity when near target)
        stability_reward = 0.0
        if len(self.position_velocity_history) > 0:
            recent_velocity = abs(self.position_velocity_history[-1])
            is_near_target = position_error < self.target_tolerance
            
            if is_near_target:
                # Reward stability when near target
                stability_reward = np.exp(-recent_velocity / self.stability_threshold)
                self.consecutive_stable_steps += 1 if recent_velocity < self.stability_threshold else 0
                
                # Bonus for sustained stability
                if self.consecutive_stable_steps > 10:
                    stability_reward += 0.5
            else:
                self.consecutive_stable_steps = 0
        
        # 3. Oscillation Penalty (penalize high-frequency position changes)
        oscillation_penalty = 0.0
        if len(self.position_velocity_history) >= 3:
            # Calculate velocity variance (measure of oscillation)
            recent_velocities = list(self.position_velocity_history)[-3:]
            velocity_variance = np.var(recent_velocities)
            oscillation_penalty = -min(velocity_variance / 10000.0, 1.0)  # Cap penalty
        
        # 4. Time in Tolerance Bonus
        tolerance_bonus = 0.0
        if position_error < self.target_tolerance:
            self.time_in_tolerance += 1
            # Increasing bonus for staying at target
            tolerance_bonus = min(self.time_in_tolerance / 100.0, 0.5)
        else:
            self.time_in_tolerance = 0
        
        # 5. Energy Efficiency (small penalty for high current)
        energy_penalty = -abs(current_mA) / 10000.0  # Small penalty
        
        # 6. Action Smoothness (penalize large servo changes)
        smoothness_penalty = 0.0
        if hasattr(self, 'last_action') and self.last_action is not None:
            action_change = abs(action - self.last_action)
            smoothness_penalty = -action_change / 1000.0  # Small penalty for large changes
        
        self.last_action = action
        
        # Combine all reward components
        total_reward = (
            position_reward * 2.0 +        # Primary objective: reach target
            stability_reward * 1.0 +       # Secondary: be stable
            oscillation_penalty * 1.5 +    # Important: minimize oscillation
            tolerance_bonus * 1.0 +        # Bonus: stay at target
            energy_penalty * 0.5 +         # Minor: energy efficiency
            smoothness_penalty * 0.3       # Minor: smooth actions
        )
        
        return float(total_reward)
    
    def get_oscillation_metrics(self) -> Dict[str, float]:
        """Get current oscillation and stability metrics for debugging"""
        metrics = {
            'position_error': abs(self.position_history[-1] - self.target_position) if self.position_history else 0.0,
            'velocity': self.position_velocity_history[-1] if self.position_velocity_history else 0.0,
            'velocity_variance': np.var(list(self.position_velocity_history)) if len(self.position_velocity_history) > 1 else 0.0,
            'time_in_tolerance': self.time_in_tolerance,
            'consecutive_stable_steps': self.consecutive_stable_steps,
            'target_position': self.target_position
        }
        return metrics
    
    def _is_done(self, state: np.ndarray) -> bool:
        """Check if episode should terminate early"""
        # Denormalize for safety checks
        real_state = self._denormalize_state(state)
        
        # Terminate if current is too high (safety)
        if abs(real_state[2]) > 3000:  # 3A current limit
            return True
            
        # Terminate if voltage is too low
        if real_state[3] < 5.0:  # 5V minimum
            return True
            
        return False
    
    def set_target_position(self, target: float):
        """Set target position for position control tasks"""
        self.target_position = target
        # Reset stability tracking when target changes
        self.time_in_tolerance = 0
        self.consecutive_stable_steps = 0
    
    def set_target_positions(self, targets: List[float]):
        """Set the list of possible target positions"""
        self.possible_targets = targets
    
    def set_target_change_probability(self, prob: float):
        """Set probability of changing target each episode"""
        self.target_change_probability = max(0.0, min(1.0, prob))
    
    def close(self):
        """Clean up resources"""
        self.disconnect()


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test flywheel RL environment')
    parser.add_argument('port', help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--steps', type=int, default=100, help='Number of test steps')
    args = parser.parse_args()
    
    print(f"Testing flywheel environment on {args.port}")
    
    try:
        env = FlywheelEnv(args.port)
        
        # Test reset
        print("Resetting environment...")
        state = env.reset()
        print(f"Initial state: {state}")
        
        # Test random actions
        print(f"Running {args.steps} random steps...")
        total_reward = 0
        
        for step in range(args.steps):
            # Random action
            action = np.random.uniform(env.action_space_low, env.action_space_high)
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            if step % 10 == 0:
                metrics = env.get_oscillation_metrics()
                print(f"Step {step}: action={action:.1f}, reward={reward:.3f}, "
                      f"pos_err={metrics['position_error']:.1f}, "
                      f"velocity={metrics['velocity']:.1f}, target={metrics['target_position']:.0f}")
            
            if done:
                print(f"Episode finished early at step {step}")
                break
        
        print(f"Total reward: {total_reward:.3f}")
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'env' in locals():
            env.close()