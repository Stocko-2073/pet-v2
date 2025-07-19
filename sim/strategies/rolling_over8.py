from sbx import SAC
import numpy as np
from . import pack_observation
from ergon_env2 import ErgonEnv2
import transforms3d


class RollingOver8:
  """
  This class uses various reward functions to evaluate the robot's performance:

  1. **Base Reward**: A constant reward of 1.0 for staying alive each step.

  LOG:
  - remove control effort penalty
  - add fatigue terminal condition
  - remove fatigue terminal condition
  - add pose match reward

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
      [21]    # Left Ankle 1dof
      [22]    # Right Ankle 1dof
  
  ctrl[14]:
      [0:2]   # Head controls (z-rot, x-rot)
      [2:5]   # Left arm controls (base, ext, hand)
      [5:8]   # Right arm controls (base, ext, hand)
      [8:11]  # Left leg controls (base, ext, foot)
      [11:14] # Right leg controls (base, ext, foot)
      [15]    # Left Ankle 1dof
      [16]    # Right Ankle 1dof
  
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
  observation_size=32
  action_size=14
  frames_per_action=20
  home_pose=np.array(
    [0,0,-0.115,1,0.0038399561722115,0,0,0,0,-0.881,1.59655,-1.59633,0.824582,-1.59655,1.59633,0,0,0,0,0,0,0,0]
  )
  home_ctrl=[0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0,0,0]

  prone_pose=np.array([0,-0.197995,-0.174105,0.838151,0.544667,-0.0159519,-0.0242356,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

  supine_pose=np.array([0,0.251216,-0.259594,0.706429,-0.707784,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

  max_match_pose_reward=0
  min_quat_dist=0

  deg2rad=lambda deg:deg*np.pi/180

  # Added thresholds for terminal conditions
  MAX_TILT_ANGLE=deg2rad(45)
  MIN_HEIGHT=-0.2  # meters
  MAX_ACCEL=50.0  # m/s^2

  # Added parameter to control the weight of the position reward
  POSITION_WEIGHT=0.5  # Adjust this to change how much staying near origin matters

  def init(self,total_timesteps=0):
    self.env=ErgonEnv2(self)
    self.env.data.qpos[:]=self.home_pose
    self.env.data.qvel[:]=0
    self.init_qpos=self.env.data.qpos.ravel().copy()
    self.init_qvel=self.env.data.qvel.ravel().copy()
    self.upper_body=np.zeros(8)

  def before_sim(self,action):
    self.step_number+=1

  def parametric_gaussian(self,x,mean,sigma):
    coefficient=1.0/(sigma*np.sqrt(2.0*np.pi))
    exponent=-0.5*((x-mean)/sigma)**2
    return coefficient*np.exp(exponent)

  def quaternion_distance(self,q1,q2):
    dot_product=abs(np.sum(q1*q2))  # abs handles double-cover
    dot_product=min(1.0,max(-1.0,dot_product))  # clamp to [-1,1]
    return 2.0*np.arccos(dot_product)

  def euler_distance(self,euler1,euler2):
    return np.minimum(np.abs(euler1-euler2),2*np.pi-np.abs(euler1-euler2))

  def matches_pose(self,joint_pos,pose,sigma):
    dist=(self.quaternion_distance(joint_pos[0:4],pose[3:7])+
          np.sum(self.euler_distance(joint_pos[4:],pose[7:])))
    return self.parametric_gaussian(dist,mean=0,sigma=sigma)

  def step(self,action):
    obs=self.observe()

    # Unpack observations
    joint_pos=obs['joint_pos']
    accel_data=obs['accel_data']
    chest_accel,torso_accel=accel_data[:3],accel_data[3:6]

    # Reward keeping motors near home position
    # reward+=np.sum(self.parametric_gaussian(action,mean=0,sigma=0.25))

    # Calculate reward
    reward=0.0

    # Added reward for pose match
    match_pose_reward=self.matches_pose(joint_pos,self.prone_pose,sigma=2)
    reward+=match_pose_reward
    self.max_match_pose_reward=np.maximum(match_pose_reward,self.max_match_pose_reward)

    # Added reward for prone orientation
    quat_dist=self.quaternion_distance(joint_pos[3:7],np.array(self.prone_pose[3:7]))
    self.min_quat_dist=np.minimum(quat_dist,self.min_quat_dist)
    prone_orientation_reward=self.parametric_gaussian(quat_dist,mean=0,sigma=1)
    reward+=prone_orientation_reward

    # Check terminal conditions
    terminated=self._check_terminal_conditions(joint_pos,chest_accel,torso_accel)

    # Check truncation
    truncated=self.step_number>=self.episode_len

    return pack_observation(obs),reward,terminated,truncated,{}

  def _check_terminal_conditions(self,joint_pos,chest_accel,torso_accel):
    """
    Check if episode should terminate due to failure conditions
    """
    return False

  def log(self,key,value):
    try:
      self.algorithm_instance.logger.record(key,value)
    except:
      print("Skipping logging")
      pass

  def reset(self):
    self.log("rollout/match_pose",self.max_match_pose_reward)
    self.log("rollout/quat_dist",self.min_quat_dist)
    self.step_number=0
    self.max_match_pose_reward=0
    self.min_quat_dist=3.14159

    qpos=np.array(self.supine_pose).ravel().copy()
    qvel=np.array(self.env.data.qvel).ravel().copy()

    # Body position + orientation
    noise_magnitude=0.025
    qpos[0:7]+=self.env.np_random.uniform(size=7,low=-noise_magnitude,high=noise_magnitude)

    # Angles
    noise_magnitude=1.0
    qpos[7:21]+=self.env.np_random.uniform(size=21-7,low=-noise_magnitude,high=noise_magnitude)

    # Velocities
    noise_magnitude=0.05
    qvel[0:21]+=self.env.np_random.uniform(size=21,low=-noise_magnitude,high=noise_magnitude)

    body_euler=np.array(transforms3d.euler.quat2euler(qpos[3:7],'sxyz'))
    body_euler[2]+=self.env.np_random.uniform(-np.pi,np.pi)
    qpos[3:7]=transforms3d.euler.euler2quat(*body_euler,'sxyz')

    self.env.set_state(qpos,qvel)
    return pack_observation(self.observe())

  def observe(self):
    """Get observations including accelerometer data"""
    joint_pos=np.array(self.env.data.qpos[3:])
    chest_accel=self.env.data.sensor('chest').data
    torso_accel=self.env.data.sensor('torso').data
    left_foot_accel=self.env.data.sensor('left_calf').data
    right_foot_accel=self.env.data.sensor('right_calf').data

    # Combine accelerometer data
    accel_data=np.concatenate([chest_accel,torso_accel,left_foot_accel,right_foot_accel])

    return {'joint_pos':joint_pos,'accel_data':accel_data}

  def apply_action(self,action):
    # Apply the action
    self.env.data.ctrl[2:]=action
