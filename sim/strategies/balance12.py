from sbx import SAC
from jax import numpy as np
from . import pack_observation
from ergon_env2 import ErgonEnv2


class Balance12:
  """
  This class uses various reward functions to evaluate the robot's performance:
  
  1. **Base Reward**: A constant reward of 1.0 for staying alive each step.
  2. **Action Reward**: A Gaussian reward based on the action applied. Calculated in the `step` method using `parametric_gaussian` function.
  3. **Orientation Penalty**: Penalizes the deviation from the upright position by calculating roll and pitch from quaternion orientation.
  4. **Acceleration Penalty**: Penalizes high accelerations to encourage smooth movements by evaluating the chest and torso accelerometer data.
  5. **Position Penalty**: Penalizes the distance from the origin to encourage the robot to stay near its starting position.
  6. **Low X Acceleration Reward**: Penalizes high accelerations in the x-direction to discourage side to side wobbling.

  These rewards and penalties are combined to yield the final reward for each step.
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
  """

  episode_len=1000
  step_number=0
  algorithm=SAC
  algorithm_config={'buffer_size':10000000,'use_sde':True}
  observation_size=33
  action_size=6
  frames_per_action=20
  home_pose=[0,0,-0.115,1,0.0038399561722115,0,0,0,0,-0.881,1.59655,-1.59633,0.824582,-1.59655,1.59633,0,0,0,0,0,0]
  home_ctrl=[0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0]

  deg2rad=lambda deg:deg*np.pi/180

  # Added thresholds for terminal conditions
  MAX_TILT_ANGLE=deg2rad(45)
  MIN_HEIGHT=-0.2  # meters
  MAX_ACCEL=50.0  # m/s^2

  # Added parameter to control the weight of the position reward
  POSITION_WEIGHT=0.5  # Adjust this to change how much staying near origin matters

  def init(self):
    self.env=ErgonEnv2(self)
    self.env.data.qpos[:]=self.home_pose
    self.env.data.qvel[:]=0
    self.init_qpos=self.env.data.qpos.ravel().copy()
    self.init_qvel=self.env.data.qvel.ravel().copy()

  def before_sim(self,action):
    self.step_number+=1


  def parametric_gaussian(self,x,mean,sigma):
    coefficient=1.0/(sigma*np.sqrt(2.0*np.pi))
    exponent=-0.5*((x-mean)/sigma)**2
    return coefficient*np.exp(exponent)


  def step(self,action):
    obs=self.observe()

    # Unpack observations
    joint_pos=obs['joint_pos']
    accel_data=obs['accel_data']
    chest_accel,torso_accel=accel_data[:3],accel_data[3:6]

    # Calculate reward
    reward=self._calculate_reward(joint_pos,chest_accel,torso_accel)

    # Apply Gaussian reward for each action element
    action_reward=np.sum(self.parametric_gaussian(action,mean=0,sigma=1)
    )  # Use a mean of 0 and sigma of 1 for this example
    reward+=action_reward  # Add the action reward to the current reward

    # Apply low X acceleration reward
    low_x_accel_reward = self.parametric_gaussian(chest_accel[0], mean=0, sigma=1) * 2
    reward += low_x_accel_reward

    # Apply low Y acceleration reward
    low_y_accel_reward = self.parametric_gaussian(chest_accel[1], mean=0, sigma=1) * 2
    reward += low_y_accel_reward

    # Check terminal conditions
    terminated=self._check_terminal_conditions(joint_pos,chest_accel,torso_accel)

    # Check truncation
    truncated=self.step_number>=self.episode_len

    return pack_observation(obs),reward,terminated,truncated,{}

  def _check_terminal_conditions(self,joint_pos,chest_accel,torso_accel):
    """
    Check if episode should terminate due to failure conditions
    """
    # Get quaternion orientation (w, x, y, z)
    quat=joint_pos[3:7]

    # Check robot height (z-position)
    height=joint_pos[2]
    if height<self.MIN_HEIGHT:
      return True

    # Check orientation (convert quaternion to euler angles)
    # This is a simplified check - you might want to use proper quaternion to euler conversion
    roll=np.arctan2(2.0*(quat[0]*quat[1]+quat[2]*quat[3]),1.0-2.0*(quat[1]*quat[1]+quat[2]*quat[2]))
    pitch=np.arcsin(2.0*(quat[0]*quat[2]-quat[3]*quat[1]))

    # Check if tilt is too large
    if abs(roll)>self.MAX_TILT_ANGLE or abs(pitch)>self.MAX_TILT_ANGLE:
      return True

    # Check acceleration (indicates instability or falling)
    if (np.linalg.norm(chest_accel)>self.MAX_ACCEL or np.linalg.norm(torso_accel)>self.MAX_ACCEL):
      return True

    return False

  def _calculate_reward(self,joint_pos,chest_accel,torso_accel):
    """
    Calculate the reward based on balance performance and position
    """
    # Base reward for staying alive
    reward=1.0

    # Penalize deviation from upright position
    quat=joint_pos[3:7]
    roll=np.arctan2(2.0*(quat[0]*quat[1]+quat[2]*quat[3]),1.0-2.0*(quat[1]*quat[1]+quat[2]*quat[2]))
    pitch=np.arcsin(2.0*(quat[0]*quat[2]-quat[3]*quat[1]))

    orientation_penalty=-(roll**2+pitch**2)

    # Penalize high accelerations (encourage smooth movement)
    acceleration_penalty=-(np.linalg.norm(chest_accel)+np.linalg.norm(torso_accel))*0.01

    # Add position penalty to encourage staying near origin
    x,y=joint_pos[0],joint_pos[1]  # Get x,y position
    position_penalty=-self.POSITION_WEIGHT*(x**2+y**2)  # Quadratic penalty based on distance from origin

    return reward+orientation_penalty+acceleration_penalty+position_penalty

  def reset(self):
    self.step_number=0

    noise_magnitude=0.005
    qpos=self.init_qpos+self.env.np_random.uniform(size=self.env.robot.nq,low=-noise_magnitude,high=noise_magnitude)
    qvel=self.init_qvel+self.env.np_random.uniform(size=self.env.robot.nv,low=-noise_magnitude,high=noise_magnitude)

    # Don't add noise to quaternion
    qpos[3:7]=self.init_qpos[3:7]

    self.env.set_state(qpos,qvel)
    return pack_observation(self.observe())

  def observe(self):
    """Get observations including accelerometer data"""
    joint_pos=np.array(self.env.data.qpos)
    chest_accel=self.env.data.sensor('chest').data
    torso_accel=self.env.data.sensor('torso').data
    left_foot_accel=self.env.data.sensor('left_foot').data
    right_foot_accel=self.env.data.sensor('right_foot').data

    # Combine accelerometer data
    accel_data=np.concatenate([chest_accel,torso_accel,left_foot_accel,right_foot_accel])

    return {'joint_pos':joint_pos,'accel_data':accel_data}

  def apply_action(self,action):
    action/=2
    # Apply the action
    self.env.data.ctrl[8:]=action
