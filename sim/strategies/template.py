from sbx import SAC
from jax import numpy as np
from . import pack_observation

class Template:
  """ The first strategy we attempted """

  episode_len=1000
  step_number=0
  algorithm=SAC
  observation_size=24
  frames_per_action=20
  home_pose=[0,0,-0.115,1,0.0038399561722115,0,0,0,0,-0.881,1.59655,-1.59633,0.824582,-1.59655,1.59633,0,0,0,0,0,0]

  def init(self):
    # Initialize state vectors from model
    self.init_qpos=self.env.data.qpos.ravel().copy()
    self.init_qvel=self.env.data.qvel.ravel().copy()

  def before_sim(self,action):
    self.step_number+=1

  def step(self,action):
    obs=self.observe()

    joint_obs=obs[:21]
    accel_obs=obs[21:]

    # Calculate reward
    reward = 1

    # Terminal conditions
    done=False
    truncated=False
    return pack_observation(obs),reward,done,truncated,{}

  def reset(self):
    self.step_number=0

    # Reduced noise magnitude
    noise_magnitude=0.005  # Reduced from 0.01

    qpos=self.init_qpos+self.env.np_random.uniform(size=self.env.model.nq,low=-noise_magnitude,high=noise_magnitude)
    qvel=self.init_qvel+self.env.np_random.uniform(size=self.env.model.nv,low=-noise_magnitude,high=noise_magnitude)

    # Don't add noise to quaternion
    qpos[3:7]=self.init_qpos[3:7]

    self.env.set_state(qpos,qvel)
    return pack_observation(self.observe())

  def apply_action(self,action):
    self.env.data.ctrl[:]=action

  def observe(self):
    """Get observations including accelerometer data"""
    joint_pos=np.array(self.env.data.qpos)
    accel_data=self.env.data.sensor('chest').data
    return {'joint_pos':joint_pos,'accel_data':accel_data}
