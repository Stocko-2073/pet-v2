from sbx import SAC
from jax import numpy as np
from . import pack_observation


class Balance:
  """ A balance only strategy """

  episode_len=1000
  step_number=0
  algorithm=SAC
  observation_size=33
  frames_per_action=20
  home_pose=[0,0,-0.115,1,0.0038399561722115,0,0,0,0,-0.881,1.59655,-1.59633,0.824582,-1.59655,1.59633,0,0,0,0,0,0]
  home_ctrl=[0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0]

  # Added thresholds for terminal conditions
  MAX_TILT_ANGLE=0.5  # radians (~28.6 degrees)
  MIN_HEIGHT=-0.2  # meters
  MAX_ACCEL=50.0  # m/s^2

  def init(self):
    self.env.data.qpos[:]=self.home_pose
    self.env.data.qvel[:]=0
    self.init_qpos=self.env.data.qpos.ravel().copy()
    self.init_qvel=self.env.data.qvel.ravel().copy()

  def before_sim(self,action):
    self.step_number+=1
    # Clear out upper body movement
    action[:9]=self.home_ctrl[:9]

  def step(self,action):
    obs=self.observe()

    # Unpack observations
    joint_pos=obs['joint_pos']
    accel_data=obs['accel_data']
    chest_accel,torso_accel=accel_data[:3],accel_data[3:6]

    # Calculate reward
    reward=self._calculate_reward(joint_pos,chest_accel,torso_accel)

    # Check terminal conditions
    done=self._check_terminal_conditions(joint_pos,chest_accel,torso_accel)

    # Check truncation
    truncated=self.step_number>=self.episode_len

    return pack_observation(obs),reward,done,truncated,{}

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
    Calculate the reward based on balance performance
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

    return reward+orientation_penalty+acceleration_penalty

  def reset(self):
    self.step_number=0

    noise_magnitude=0.005
    qpos=self.init_qpos+self.env.np_random.uniform(size=self.env.model.nq,low=-noise_magnitude,high=noise_magnitude)
    qvel=self.init_qvel+self.env.np_random.uniform(size=self.env.model.nv,low=-noise_magnitude,high=noise_magnitude)

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
    action[:9]=self.home_ctrl[:9]
    self.env.data.ctrl[:]=action
