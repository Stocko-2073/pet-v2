from abc import ABC, abstractmethod

class Env(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def observation_shape(self):
    pass

  @abstractmethod
  def action_shape(self):
    pass

  @abstractmethod
  def step(self):
    pass

  @abstractmethod
  def reset(self):
    pass
