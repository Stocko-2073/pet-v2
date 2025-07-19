from sbx import SAC
from jax import numpy as np
from . import pack_observation
from ergon_env2 import ErgonEnv2

import numpy as np

import numpy as np

class BalancingHomeostat:
  def __init__(self, num_motors: int, threshold: float = 0.5):
    self.input_size = 3
    self.output_size = num_motors
    self.num_units = 8 * num_motors

    # State variables
    self.X = np.zeros(self.num_units)

    # Internal weights
    self.W = np.random.uniform(-1, 1, (self.num_units, self.num_units))

    # Sensor weights
    self.S = np.random.uniform(-1, 1, (self.num_units, self.input_size))

    # Output weights
    self.O = np.random.uniform(-1, 1, (self.output_size, self.num_units))

    # Target accelerometer reading when upright
    self.target = np.array([0.0, 0.0, 1.0])

    # Parameters
    self.threshold = threshold
    self.dt = 0.01

  def step(self, accel: np.ndarray) -> np.ndarray:
    # Normalize accelerometer input
    accel_normalized = accel / np.linalg.norm(accel)

    # Calculate error from upright position
    error = accel_normalized - self.target

    # Use error magnitude to influence stability threshold
    error_magnitude = np.linalg.norm(error)
    #print(accel_normalized, self.target, error_magnitude)

    current_threshold = self.threshold * (1.0 - 0.5 * error_magnitude)

    # Input includes both current reading and error
    full_input = np.concatenate([accel_normalized, error])

    # Adjust sensor weights matrix to accommodate error input
    if self.S.shape[1] != len(full_input):
      self.S = np.random.uniform(-1, 1, (self.num_units, len(full_input)))

    # Calculate new state
    dX = -self.X + self.W @ self.X + self.S @ full_input
    self.X += self.dt * dX

    norm_X = np.linalg.norm(self.X)
    if norm_X > 1.0:
      self.X = self.X / norm_X

    # System is unstable if either internal states exceed threshold
    # OR if we're far from upright position
    unstable = (np.any(np.abs(self.X) > current_threshold) or
                error_magnitude > 0.5)  # Adjust this threshold as needed

    # If unstable, randomly adjust weights
    if unstable:
      for i in range(self.num_units):
        if np.abs(self.X[i]) > current_threshold or error_magnitude > 0.5:
          # Adjust weights with bias towards reducing error
          w_adjust = np.random.uniform(-0.1, 0.1, self.num_units)
          s_adjust = np.random.uniform(-0.1, 0.1, len(full_input))

          # Add bias to push towards upright position
          if error_magnitude > 0.5:
            error_bias = np.concatenate([-error, -error])  # 6D bias
            s_adjust += 0.05 * error_bias

          self.W[i] += w_adjust
          self.S[i] += s_adjust

          # Clip weights
          self.W[i] = np.clip(self.W[i], -1, 1)
          self.S[i] = np.clip(self.S[i], -1, 1)

    # Calculate motor outputs
    outputs = self.O @ self.X

    # Clip outputs
    outputs = np.clip(outputs, -1, 1)

    outputs *= 1.5  # Scale to actuator range

    return outputs

class Homeostat1:
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

  episode_len=1000
  total_steps=0
  step_number=0
  algorithm=SAC
  algorithm_config={
    'buffer_size':10000000,
    'use_sde':True,
  }
  observation_size=14+12
  action_size=6
  frames_per_action=4
  home_pose=[0,0,-0.115,1,0.0038399561722115,0,0,0,0,-0.881,1.59655,-1.59633,0.824582,-1.59655,1.59633,0,0,0,0,0,0]
  home_ctrl=[0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0]

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
    self.homeostat = BalancingHomeostat(num_motors=12)

  def before_sim(self,action):
      self.step_number+=1

  def parametric_gaussian(self,x,mean,sigma):
    return np.exp(-((x-mean)*np.pi/sigma)**2)

  def step(self,action):
    # Not used
    pass


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
      size=13,  #
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
    left_foot_accel=self.env.data.sensor('left_foot').data
    right_foot_accel=self.env.data.sensor('right_foot').data

    # Combine accelerometer data
    accel_data=np.concatenate([chest_accel,torso_accel,left_foot_accel,right_foot_accel])

    return {'joint_pos':joint_pos,'accel_data':accel_data}

  def apply_action(self,action):
    self.total_steps+=1
    # if self.total_steps%100==0:
    #   # Randomize upper body
    #   self.upper_body=self.env.np_random.uniform(size=8,low=-2.156,high=2.156)
    # self.env.data.ctrl[:8]=self.upper_body
    self.env.data.ctrl[:2]=self.home_ctrl[:2]
    # self.env.data.ctrl[8:]=action
    accel = self.env.data.sensor('chest').data
    outputs = self.homeostat.step(accel)
    self.env.data.ctrl[2:] = outputs
