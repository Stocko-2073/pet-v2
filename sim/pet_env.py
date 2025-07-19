from jax import numpy as np
from gymnasium import utils,Env
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import sys
import mujoco

class PetEnv(Env,utils.EzPickle):
  """
  Notes About Robot Configuration:

  qpos[21]:
      [0:3]   # Base position (x, y, z)
      [3:7]   # Base quaternion (w, x, y, z)
      [7:10]  # Leg 1 (hip, knee, foot)
      [10:13] # Leg 2 (hip, knee, foot)
      [13:16] # Leg 3 (hip, knee, foot)

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

  def __init__(self,strategy,episode_len=1500,**kwargs):
    utils.EzPickle.__init__(self,**kwargs)
    Env.__init__(self)
    self.strategy=strategy

    model_path=os.path.abspath("../pet/world.xml")
    self.robot=mujoco.MjModel.from_xml_path(model_path)
    self.data=mujoco.MjData(self.robot)

    self.observation_space=Box(low=-np.inf,high=np.inf,shape=(strategy.observation_size,),dtype=np.float32)
    self.action_space=Box(low=-1,high=1,shape=(strategy.action_size,),dtype=np.float32)

    self.metadata={"render_modes":["human","rgb_array","depth_array",],"render_fps":1000/strategy.frames_per_action,
    }

    self.episode_len=episode_len

  def step(self,action):
    self.strategy.before_sim(action)
    self.strategy.apply_action(action)
    mujoco.mj_step(self.robot,self.data,nstep=self.strategy.frames_per_action)
    return self.strategy.step(action)

  def reset(self,seed=0):
    return self.strategy.reset(),[]

  def set_state(self,qpos,qvel):
    self.data.qpos[:]=qpos
    self.data.qvel[:]=qvel
