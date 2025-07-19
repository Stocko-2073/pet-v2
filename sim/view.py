import argparse
import mujoco
import mujoco.viewer
import time
import sys
import importlib
from pathlib import Path

from sbx import SAC, PPO

from strategies import pack_observation

snake_to_pascal=lambda s:''.join(word.capitalize() for word in s.split('_'))

class ViewingSession:
  def __init__(self,strategy_name,model_path):
    self.strategy_name=strategy_name
    if model_path is not None:
      self.model_path=Path(model_path)

      if not self.model_path.exists():
        raise FileNotFoundError(f"Model file {self.model_path} not found")

    # Initialize strategy and environment
    strategy=self.instantiate_strategy(strategy_name)
    strategy.init()
    self.env=strategy.env
    if model_path is not None:
      self.model=strategy.algorithm.load(self.model_path)
    else:
      self.model=None

  def instantiate_strategy(self,strategy_name):
    strategy_module_path=f"strategies.{strategy_name}"
    strategy_class_name=snake_to_pascal(strategy_name)
    strategy_module=importlib.import_module(strategy_module_path)
    strategy_class=getattr(strategy_module,strategy_class_name)
    return strategy_class()

  def key_callback(self,key):
    if key==ord("R"):
      print("Resetting environment")
      self.env.strategy.reset()

  def run(self):
    with mujoco.viewer.launch_passive(self.env.robot,self.env.data,key_callback=self.key_callback) as viewer:
      # Configure viewer settings
      with viewer.lock():
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True

      # Main viewing loop
      mujoco.mj_step(self.env.robot,self.env.data)
      while viewer.is_running():
        step_start=time.time()

        # Get observation and predict action
        obs=pack_observation(self.env.strategy.observe())
        if self.model is not None:
          action,_states=self.model.predict(obs,deterministic=True)
          self.env.strategy.apply_action(action)
        else:
          self.env.strategy.apply_action(None)
        for _ in range(self.env.strategy.frames_per_action):
          mujoco.mj_step(self.env.robot,self.env.data)
          viewer.sync()

        # Time keeping
        time_until_next_step=(self.env.strategy.frames_per_action*self.env.robot.opt.timestep-(time.time()-step_start))
        if time_until_next_step>0:
          time.sleep(time_until_next_step)


if __name__=="__main__":
  parser=argparse.ArgumentParser(description="View a trained reinforcement learning model in the Ergon environment.")
  parser.add_argument("strategy",type=str,help="Strategy used for the model.")
  parser.add_argument("model_path",type=str,nargs='?',default=None,help="Path to the trained model file.")

  args=parser.parse_args()

  session=ViewingSession(strategy_name=args.strategy,model_path=args.model_path)
  session.run()
