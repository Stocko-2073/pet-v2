from sbx import SAC
from jax import numpy as np

from ergon_env2 import ErgonEnv2
from . import pack_observation,Affine
import transforms3d as t3d


class Delta:
  """
  Notes About Robot Configuration:
  
  qpos[21]:
      [0:3]   # Base position (x, y, z)
      [3:7]   # Base quaternion (w, x, y, z)
      [7:9]   # Head (z-rot, x-rot)
      [9:12]  # Left arm (shoulder, elbow, hand)
      [12:15] # Right arm (shoulder, elbow, hand)
      [15:18] # Left leg (hip, knee, foot)
      [18:21] # Right leg (hip, knee, foot)
  
  ctrl[14]:
      [0:2]   # Head controls (z-rot, x-rot)
      [2:5]   # Left arm controls (base, ext, hand)
      [5:8]   # Right arm controls (base, ext, hand)
      [8:11]  # Left leg controls (hip, knee, foot)
      [11:14] # Right leg controls (hip, knee, foot)
  
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
  action_size=14
  frames_per_action=20

  balanced_pose=[0,0,-0.1175,  # Base position (x, y, z)
    1,0.0038399561722115,0,0,  # Base quaternion (w, x, y, z)
    0,0,  # Head (z-rot, x-rot)
    -0.881,1.59655,-1.59633,  # Left arm (shoulder, elbow, hand)
    0.824582,-1.59655,1.59633,  # Right arm (shoulder, elbow, hand)
    0,0,0,  # Left leg (hip, knee, foot)
    0,0,0  # Right leg (hip, knee, foot)
  ]

  home_ctrl=[0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0]

  deg2rad=lambda deg:deg*np.pi/180
  # Added thresholds for terminal conditions
  MAX_TILT_ANGLE=0.5  # deg2rad(45)
  MIN_HEIGHT=-0.2  # meters
  MAX_ACCEL=50.0  # m/s^2

  # Added parameter to control the weight of the position reward
  POSITION_WEIGHT=0.5  # Adjust this to change how much staying near origin matters
  CTRL_STEP=0.1

  def init(self):
    self.env=ErgonEnv2(self)
    self.env.data.qpos[:]=self.balanced_pose
    self.env.data.qvel[:]=0
    self.init_qpos=self.env.data.qpos.ravel().copy()
    self.init_qvel=self.env.data.qvel.ravel().copy()

  def before_sim(self,action):
    self.step_number+=1

  def step(self,action):
    obs=self.observe()

    # Unpack observations
    joint_pos=obs['joint_pos']
    chest_accel=obs['chest_accel']
    torso_accel=obs['torso_accel']
    left_foot_accel=obs['left_foot_accel']
    right_foot_accel=obs['right_foot_accel']

    # Calculate reward
    reward=self._calculate_reward(joint_pos,chest_accel,torso_accel)

    # Check terminal conditions
    terminated=self._check_terminal_conditions(joint_pos,chest_accel,torso_accel)

    # Check truncation
    truncated=self.step_number>=self.episode_len

    return pack_observation(obs),reward,terminated,truncated,{}

  def _check_terminal_conditions(self,joint_pos,chest_accel,torso_accel):
    """
    Check if episode should terminate due to failure conditions
    """
    # Check robot height (z-position)
    # height=joint_pos[2]
    # if height<self.MIN_HEIGHT:
    #   return True

    # Check orientation (convert quaternion to euler angles)
    # This is a simplified check - you might want to use proper quaternion to euler conversion
    euler=t3d.euler.quat2euler(joint_pos[3:7],axes='sxyz')
    roll,pitch=euler[0],euler[1]

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
    euler=t3d.euler.quat2euler(joint_pos[3:7],axes='sxyz')
    roll,pitch=euler[0],euler[1]

    orientation_penalty=-(roll**2+pitch**2)

    # Penalize high accelerations (encourage smooth movement)
    acceleration_penalty=-(np.linalg.norm(chest_accel)+np.linalg.norm(torso_accel))*0.01

    # Add position penalty to encourage staying near origin
    # balance_point=Affine()
    # balance_point.translate(joint_pos[0],joint_pos[1],joint_pos[2])
    # balance_point.rotate(joint_pos[3:7])
    # balance_point.translate(0,0,0.258)
    # x,y=t3d.affines.decompose(balance_point.matrix)[0][:2]
    x,y=joint_pos[0],joint_pos[1]
    position_penalty=-self.POSITION_WEIGHT*(x**2+y**2)  # Quadratic penalty based on distance from origin

    # Try and align to the balance pose
    # Calculate shaped rewards using separate potential functions.
    # Each potential function gets its own shaping F(s,s') = γΦ(s') - Φ(s)
    # for pos in joint_pos:
    #   phi = abs(pos - self.balanced_pose)
    #   gamma = 0.1
    #   prev_phi = abs(self.prev_joint_pos - self.balanced_pose)
    #   penalty = gamma*phi - prev_phi
    #   reward+=gamma*phi-prev_phi
    # self.prev_joint_pos = joint_pos

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
    return {'joint_pos':joint_pos,'chest_accel':chest_accel,'torso_accel':torso_accel,'left_foot_accel':left_foot_accel,
      'right_foot_accel':right_foot_accel
    }

  def apply_action(self,action):
    # Clear out upper body movement
    action[:8]=self.home_ctrl[:8]

    # Apply the action
    self.env.data.ctrl[:]=action
