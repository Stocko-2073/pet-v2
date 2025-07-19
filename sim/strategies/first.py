from stable_baselines3.sac import SAC
from jax import numpy as np
from pet_env import PetEnv

class First:
    """ The first strategy we attempted """

    episode_len=1000
    step_number=0
    algorithm=SAC
    algorithm_config={'buffer_size':10000000,'use_sde':True}
    observation_size=19
    action_size=9
    frames_per_action=20
    home_pose=[0,0,-0.115,1,0.0038399561722115,0,0,0,0,-0.881,1.59655,-1.59633,0.824582,-1.59655,1.59633,0,0,0,0,0,0,0,0]
    home_ctrl=[0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0,0,0]
    ctrl_cost_weight = 0.1
    previous_action = None

    deg2rad=lambda deg:deg*np.pi/180
    pose = [0, 0, -0.115, 1, 0.0038399561722115, 0, 0, 0, 0, -0.881,
            1.59655, -1.59633, 0.824582, -1.59655, 1.59633, 0]

    def init(self, total_timesteps=0):
        self.env=PetEnv(self)
        self.env.data.qpos[:]=0
        self.env.data.qvel[:]=0

        # Initialize state vectors from model
        self.init_qpos = self.env.data.qpos.ravel().copy()
        self.init_qvel = self.env.data.qvel.ravel().copy()

        # Initialize weights for different joint groups
        self.joint_weights = np.ones(self.observation_size-3)

    def before_sim(self, action):
        self.step_number += 1

    def step(self, action):
        obs = self._get_obs()
        joint_obs = obs[:16]
        accel_obs = obs[16:]
        # print(accel_obs)

        # Simple RL reward and termination logic for a walking robot

        # Unpack joint and accelerometer observations
        # joint_obs: [0:16] (positions, orientations, joint angles, etc.)
        # accel_obs: [16:19] (accelerometer x, y, z)

        # Reward for moving forward (positive x direction)
        x_pos = joint_obs[0]
        x_vel = self.env.data.qvel[0]  # forward velocity

        # Encourage forward progress based on distance traveled
        distance_traveled = x_pos - self.initial_x_pos
        forward_reward = distance_traveled

        # Penalize deviation from upright orientation (quaternion in joint_obs[3:7])
        quat = joint_obs[3:7]
        # Calculate roll and pitch from quaternion
        roll = np.arctan2(2.0*(quat[0]*quat[1]+quat[2]*quat[3]), 1.0-2.0*(quat[1]**2+quat[2]**2))
        pitch = np.arcsin(2.0*(quat[0]*quat[2]-quat[3]*quat[1]))
        upright_penalty = -(roll**2 + pitch**2)

        # Total reward
        reward = forward_reward + upright_penalty

        # Termination conditions
        # If robot falls (z too low or tilts too much)
        z = joint_obs[2]
        max_tilt = First.deg2rad(45)
        done = False
        if abs(roll) > max_tilt or abs(pitch) > max_tilt:
            done = True


        truncated = self.step_number > self.episode_len
        return obs, reward, done, truncated, {}

    def reset(self):
        self.step_number = 0

        # Reduced noise magnitude
        noise_magnitude = 0.005  # Reduced from 0.01

        qpos = self.init_qpos + self.env.np_random.uniform(
            size=len(self.init_qpos),
            low=-noise_magnitude,
            high=noise_magnitude)
        qvel = self.init_qvel + self.env.np_random.uniform(
            size=len(self.init_qvel),
            low=-noise_magnitude,
            high=noise_magnitude)

        # Don't add noise to quaternion
        qpos[3:7] = self.init_qpos[3:7]

        self.env.set_state(qpos, qvel)
        
        # Track initial x position for distance-based reward
        self.initial_x_pos = qpos[0]
        
        return self._get_obs()

    def get_accelerometer_data(self):
        """Get the current accelerometer readings"""
        return self.env.data.sensor('body').data

    def _get_obs(self):
        """Get observations including accelerometer data"""
        # Get joint positions
        joint_pos = np.array(self.env.data.qpos)

        # Get accelerometer readings
        accel_data = self.get_accelerometer_data()

        # Combine joint positions and accelerometer data
        obs = np.concatenate([joint_pos, accel_data])

        return obs

    def observe(self):
        joint_pos = np.array(self.env.data.qpos)
        accel_data = self.get_accelerometer_data()
        return {'joint_pos':joint_pos,'accel_data':accel_data}

    def apply_action(self,action):
        self.env.data.ctrl[:]=action
