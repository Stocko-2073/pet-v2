import argparse
import os
import sys
import time
import random
import importlib
from pathlib import Path

from pet_env import PetEnv
from stable_baselines3.common.env_checker import check_env

snake_to_pascal=lambda s:''.join(word.capitalize() for word in s.split('_'))


class TrainingSession:
  def __init__(self,strategy_name=None,id=0,render_mode="rgb_array",tensorboard_log="../tensorboard/",
      total_timesteps=100000,checkpoint_interval=50000
  ):

    self.start_time=time.time()

    # Prepare checkpoint directory
    self.datestamp=time.strftime("%Y%m%d-%H%M%S")
    self.checkpoint_dir=Path("..","checkpoints",strategy_name,self.datestamp)
    os.makedirs(self.checkpoint_dir,exist_ok=True)

    # Prepare models directory
    self.models_dir=Path("..","models",strategy_name)
    os.makedirs(self.models_dir,exist_ok=True)

    # Prepare tensorboard log directory
    tensorboard_log=Path(tensorboard_log,strategy_name,self.datestamp)

    # Instatiate strategy and environment
    self.strategy_name=strategy_name
    strategy=self.instantiate_strategy(strategy_name)
    strategy.parallel_id=id
    strategy.init(total_timesteps)
    if strategy.algorithm is None:
      raise NotImplementedError(f"Strategy {strategy.__class__.__name__} does not have an algorithm defined")
    self.algorithm=strategy.algorithm("MlpPolicy",strategy.env,verbose=1,device="mps",tensorboard_log=tensorboard_log,
      **(strategy.algorithm_config or {})
    )
    strategy.algorithm_instance = self.algorithm

    self.total_timesteps=total_timesteps
    self.checkpoint_interval=checkpoint_interval

  def instantiate_strategy(self,strategy_name):
    strategy_module_path=f"strategies.{strategy_name}"
    strategy_class_name=snake_to_pascal(strategy_name)
    strategy_module=importlib.import_module(strategy_module_path)
    strategy_class=getattr(strategy_module,strategy_class_name)
    return strategy_class()

  def train(self):
    start_time=time.time()
    for step in range(0,self.total_timesteps,self.checkpoint_interval):
      if step>0:
        self.algorithm.save(Path(self.checkpoint_dir,f"checkpoint_{self.strategy_name}_{step}"))
        print(f"Saved checkpoint at {step} steps")
      self.algorithm.learn(total_timesteps=self.checkpoint_interval,reset_num_timesteps=False,log_interval=10)
      print(f"Wall time: {time.time()-self.start_time:.2f} | Steps: {step+self.checkpoint_interval}")
    self.algorithm.save(Path(self.models_dir,f"pet_{self.strategy_name}_{self.datestamp}"))
    self.algorithm.save(Path(self.models_dir,f"pet_{self.strategy_name}_latest"))
    print("Training finished and final model saved")
    total_training_time_minutes=(time.time()-start_time)/60
    print(f"Total training time: {total_training_time_minutes:.2f} minutes")


if __name__=="__main__":
  parser=argparse.ArgumentParser(description="Train a reinforcement learning agent for the Pet environment.")
  parser.add_argument("strategy",type=str,help="Strategy to be used for training.")
  parser.add_argument("--id",type=int,default=0,help="ID of parallel training sessions.")
  parser.add_argument("--timesteps",type=int,default=2000000,help="Total timesteps for training.")
  parser.add_argument("--interval",type=int,default=50000,help="Interval of steps at which to save checkpoints.")

  args=parser.parse_args()

  session=TrainingSession(strategy_name=args.strategy,id=args.id,total_timesteps=args.timesteps,
    checkpoint_interval=args.interval
  )
  session.train()
