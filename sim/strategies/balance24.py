from sbx import SAC
from jax import numpy as np
from jax import jit
from . import pack_observation
from ergon_env2 import ErgonEnv2

@jit
def parametric_gaussian(x,mean,sigma):
  return np.exp(-((x-mean)*np.pi/sigma)**2)

@jit
def calc_reward(obs):
  # Unpack observations
  joint_pos=obs['joint_pos']
  accel_data=obs['accel_data']
  chest_accel,torso_accel=accel_data[:3],accel_data[3:6]
  left_foot_accel,right_foot_accel=accel_data[6:9],accel_data[9:12]
  # Calculate reward
  reward=0.1  # Base reward for staying alive
  reward+=parametric_gaussian(chest_accel[2],mean=9.8,sigma=1)
  reward+=parametric_gaussian(torso_accel[2],mean=9.8,sigma=1)
  reward+=parametric_gaussian(left_foot_accel[2],mean=9.8,sigma=1)
  reward+=parametric_gaussian(right_foot_accel[2],mean=9.8,sigma=1)
  # Apply Gaussian reward for each action element
  # scale = 1.0 / len(action)
  # action_reward=np.sum(self.parametric_gaussian(action,mean=0,sigma=0.25) * scale)
  # reward+=action_reward  # Add the action reward to the current reward
  # Apply low X acceleration reward
  # low_x_accel_reward=self.parametric_gaussian(chest_accel[0],mean=0,sigma=1)
  reward+=parametric_gaussian(chest_accel[0],mean=0,sigma=1)
  reward+=parametric_gaussian(torso_accel[0],mean=0,sigma=1)
  reward+=parametric_gaussian(left_foot_accel[0],mean=0,sigma=1)
  reward+=parametric_gaussian(right_foot_accel[0],mean=0,sigma=1)
  # Apply low Y acceleration reward
  # low_y_accel_reward=self.parametric_gaussian(chest_accel[1],mean=0,sigma=1)
  reward+=parametric_gaussian(chest_accel[1],mean=0,sigma=1)
  reward+=parametric_gaussian(torso_accel[1],mean=0,sigma=1)
  reward+=parametric_gaussian(left_foot_accel[1],mean=0,sigma=1)
  reward+=parametric_gaussian(right_foot_accel[1],mean=0,sigma=1)
  # num_rewards=5
  # reward /= num_rewards
  # Penalize high accelerations (encourage smooth movement)
  # acceleration_penalty=-(np.linalg.norm(chest_accel)+np.linalg.norm(torso_accel))*0.01
  # reward += acceleration_penalty
  # Check terminal conditions
  return reward


class Balance24:
  """
  This class uses various reward functions to evaluate the robot's performance:
  
  1. **Base Reward**: A constant reward of 1.0 for staying alive each step.
  2. **Action Reward**: A Gaussian reward based on the action applied. Calculated in the `step` method using `parametric_gaussian` function.
  3. **Orientation Penalty**: Penalizes the deviation from the upright position by calculating roll and pitch from quaternion orientation.
  4. **Acceleration Penalty**: Penalizes high accelerations to encourage smooth movements by evaluating the chest and torso accelerometer data.
  5. **Position Penalty**: Penalizes the distance from the origin to encourage the robot to stay near its starting position.
  6. **Low X Acceleration Reward**: Penalizes high accelerations in the x-direction to discourage side to side wobbling.

  These rewards and penalties are combined to yield the final reward for each step.

  LOG:
  - Adjusted action reward sigma to 0.5
  - Adjusted action reward sigma to 0.25
  - Increased noise magnitude by a factor of 10 to 0.05
  - 10x was too tough, training fell below baseline.
  - Changed noise magnitude to a factor of 5 to 0.025
  - Changed noise magnitude back to 0.005
  - Randomize upper body controls when resetting
  - Do not reset upper body joint poses
  - Add total steps and reset upper body controls at longer term intervals
  - Uses gaussian reward for position and orientation penalties
  - Randomize upper body every 100 steps
  - Reworked observations to not include robot position and real-time joint info
  - Adjusting base reward to 0.1
  - Adjusting sigma for chest_accel_reward to 0.25

  """

  """
  Notes About Robot Configuration:
  
  qpos[21]:
      [0:3]   # Base position (x, y, z)
      [3:7]   # Base quaternion (w, x, y, z)
      [7:9]   # Head (z-rot, x-rot)
      [9:12]  # Left arm (base, ext, hand)
      [12:15] # Right arm (base, ext, hand)
      [15:18] # Left leg (base, ext, foot)
      [18:21] # Right leg (base, ext, foot)
  
  ctrl[14]:
      [0:2]   # Head controls (z-rot, x-rot)
      [2:5]   # Left arm controls (base, ext, hand)
      [5:8]   # Right arm controls (base, ext, hand)
      [8:11]  # Left leg controls (base, ext, foot)
      [11:14] # Right leg controls (base, ext, foot)
  
  Note: All actuators limited to [-2.156, 2.156] range
  
  <accelerometer name="chest" site="chest_site" />
  <accelerometer name="torso" site="torso_site" />
  <accelerometer name="left_foot" site="left_foot_site" />
  <accelerometer name="right_foot" site="right_foot_site" />
  <gyro name="chest_gyro" site="chest_site" />
  
  """

  episode_len=100
  total_steps=0
  step_number=0
  algorithm=SAC
  algorithm_config={
    'buffer_size':10000000,
    'use_sde':True,
  }
  observation_size=16+12
  action_size=8
  frames_per_action=20
  home_pose=[0,0,-0.115,1,0.0038399561722115,0,0,0,0,-0.881,1.59655,-1.59633,0.824582,-1.59655,1.59633,0,0,0,0,0,0,0,0]
  home_ctrl=[0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0,0,0]

  deg2rad=lambda deg:deg*np.pi/180

  # Added thresholds for terminal conditions
  MAX_TILT_ANGLE=deg2rad(45)
  MIN_HEIGHT=-0.2  # meters
  MAX_ACCEL=50.0  # m/s^2
  RAND_UPPER_BODY_AFTER=10000  # Randomize upper body controls every N steps

  # Added parameter to control the weight of the position reward
  POSITION_WEIGHT=0.5  # Adjust this to change how much staying near origin matters

  def init(self):
    self.env=ErgonEnv2(self)
    self.env.data.qpos[:]=self.home_pose
    self.env.data.qvel[:]=0
    self.init_qpos=self.env.data.qpos.ravel().copy()
    self.init_qvel=self.env.data.qvel.ravel().copy()
    self.upper_body=np.zeros(8)

  def before_sim(self,action):
    self.step_number+=1

  def step(self,action):
    obs=self.observe()

    reward=calc_reward(obs)

    terminated=self._check_terminal_conditions()
    # terminated = self.step_number > (self.total_steps/500+1) and reward < 0
    # Check truncation
    truncated=self.step_number>=self.episode_len
    return pack_observation(obs),reward,terminated,truncated,{}

  def _check_terminal_conditions(self):
    """
    Check if episode should terminate due to failure conditions
    """
    # Check orientation (convert quaternion to euler angles)
    # This is a simplified check - you might want to use proper quaternion to euler conversion
    quat=self.env.data.qpos[3:7]
    roll=np.arctan2(2.0*(quat[0]*quat[1]+quat[2]*quat[3]),1.0-2.0*(quat[1]*quat[1]+quat[2]*quat[2]))
    pitch=np.arcsin(2.0*(quat[0]*quat[2]-quat[3]*quat[1]))

    # Check if tilt is too large
    if abs(roll)>self.MAX_TILT_ANGLE or abs(pitch)>self.MAX_TILT_ANGLE:
      return True

    # Check acceleration (indicates instability or falling)
    chest_accel=self.env.data.sensor('chest').data
    torso_accel=self.env.data.sensor('torso').data
    if (np.linalg.norm(chest_accel)>self.MAX_ACCEL or np.linalg.norm(torso_accel)>self.MAX_ACCEL):
      return True

    return False

  def reset(self):
    self.step_number=0

    noise_magnitude=0.005

    qpos=self.env.data.qpos.ravel().copy()
    qvel=self.env.data.qvel.ravel().copy()

    qpos[0:3]=self.init_qpos[0:3]+self.env.np_random.uniform(  #
      size=3,  #
      low=-noise_magnitude,  #
      high=noise_magnitude
    )
    # Don't add noise to quaternion
    qpos[3:7]=self.init_qpos[3:7]
    qpos[15:21]=self.init_qpos[15:21]+self.env.np_random.uniform(  #
      size=6,  #
      low=-noise_magnitude,  #
      high=noise_magnitude
    )

    qvel[0:3]=self.init_qvel[0:3]+self.env.np_random.uniform(  #
      size=3,  #
      low=-noise_magnitude,  #
      high=noise_magnitude
    )
    qvel[3:7]=self.init_qvel[3:7]
    qvel[7:21]=self.init_qvel[7:21]+self.env.np_random.uniform(  #
      size=14,  #
      low=-noise_magnitude,  #
      high=noise_magnitude
    )

    self.env.set_state(qpos,qvel)
    return pack_observation(self.observe())

  def observe(self):
    """Get observations including accelerometer data"""
    joint_pos=np.array(self.env.data.ctrl)
    chest_accel=self.env.data.sensor('chest').data
    torso_accel=self.env.data.sensor('torso').data
    left_foot_accel=self.env.data.sensor('left_calf').data
    right_foot_accel=self.env.data.sensor('right_calf').data

    # Combine accelerometer data
    accel_data=np.concatenate([chest_accel,torso_accel,left_foot_accel,right_foot_accel])

    return {'joint_pos':joint_pos,'accel_data':accel_data}

  def apply_action(self,action):
    self.total_steps+=1
    # if self.total_steps%100==0:
    #   # Randomize upper body
    #   self.upper_body=self.env.np_random.uniform(size=8,low=-2.156,high=2.156)
    # self.env.data.ctrl[:8]=self.upper_body
    self.env.data.ctrl[:8]=self.home_ctrl[:8]
    self.env.data.ctrl[8:]=action
