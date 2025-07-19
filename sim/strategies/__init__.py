from jax import numpy as np

def pack_observation(obs):
  """Pack observation dictionary into a single array"""
  parts = []
  for key in sorted(obs.keys()):
    parts.append(obs[key])
  return np.concatenate(parts)

# 3D matrix functions
class Affine:
  def __init__(self, matrix=None):
    if matrix is None:
      self.matrix = np.eye(4)
    else:
      self.matrix = matrix.copy()

  def identity():
    return np.eye(4)

  def translate(self, x, y, z):
    return Affine(np.dot(self.matrix, np.array([
      [1, 0, 0, x],
      [0, 1, 0, y],
      [0, 0, 1, z],
      [0, 0, 0, 1]])))

  def scale(self, x, y, z):
    return Affine(np.dot(self.matrix, np.array([
      [x, 0, 0, 0],
      [0, y, 0, 0],
      [0, 0, z, 0],
      [0, 0, 0, 1]])))

  def xrot(self, angle):
    return Affine(np.dot(self.matrix, np.array([
      [1, 0, 0, 0],
      [0, np.cos(angle), -np.sin(angle), 0],
      [0, np.sin(angle), np.cos(angle), 0],
      [0, 0, 0, 1]])))

  def yrot(self, angle):
    return Affine(np.dot(self.matrix, np.array([
      [np.cos(angle), 0, np.sin(angle), 0],
      [0, 1, 0, 0],
      [-np.sin(angle), 0, np.cos(angle), 0],
      [0, 0, 0, 1]])))

  def zrot(self, angle):
    return Affine(np.dot(self.matrix, np.array([
      [np.cos(angle), -np.sin(angle), 0, 0],
      [np.sin(angle), np.cos(angle), 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]])))

  def rotate(self, quaternion):
    x, y, z, w = quaternion
    rotation_matrix=np.array([
      [1-2*(y**2+z**2),2*(x*y-z*w),2*(x*z+y*w),0],
      [2*(x*y+z*w),1-2*(x**2+z**2),2*(y*z-x*w),0],
      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x**2+y**2),0],
      [0,0,0,1]])
    return Affine(np.dot(self.matrix,rotation_matrix))
