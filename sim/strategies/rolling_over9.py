from sbx import SAC
import numpy as np
from . import pack_observation
from ergon_env2 import ErgonEnv2
import transforms3d


class RollingOver9:
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
    self.env.data.qpos[:]=self.supine_pose
    self.seq_index=0
    self.curr_time=0
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

  script = [
    {'time':10, 'ctrl':[0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0,0,0]},
    {'time':5, 'ctrl':[0,0,0.7,2.16,1.57,-0.7,-2.16,-1.57,0,0,0,0,0,0,0,0]},
    {'time':5, 'ctrl':[0,0,-1.57,2.16,1.57,1.57,-2.16,-1.57,0,0,0,0,0,0,0,0]},
    {'time':5, 'ctrl':[0,0,-1.57,0,1.57,1.57,0,-1.57,-1.3,0,0,0,1.3,0,0,0]},
    {'time':5, 'ctrl':[0,0,-1.57,0,1.57,1.57,0,-1.57,-1.3,0.8,-1.0,0,1.3,-0.8,1.0,0]},
    {'time':5, 'ctrl':[0,0,-1,0,0,1,0,0,0,0.0,0.1,-0.09,0,-0.0,-0.1,0.09]},
    {'time':5, 'ctrl':[0,0,-1,1.57,-1.57,1,-1.57,1.57,0,0.0,0.1,-0.09,0,-0.0,-0.1,0.09]},
  ]

  seq_index = 0
  curr_time=0
  def apply_action(self,action):
    if self.seq_index < len(self.script):
      self.env.data.ctrl[:]=self.script[self.seq_index]['ctrl']
      self.curr_time+=1
      if self.curr_time>=self.script[self.seq_index]['time']:
        self.seq_index+=1
        print("Step",self.seq_index)
        self.curr_time=0

