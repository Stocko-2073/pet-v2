import os
import random
import torch
import genesis as gs

gs.init(backend=gs.gpu)

scene=gs.Scene(
  show_viewer=True,
  viewer_options=gs.options.ViewerOptions(
    camera_pos=(3.5,-1.0,2.5),
    camera_lookat=(0.0,0.0,0.5),
    camera_fov=40,
  ),
  rigid_options=gs.options.RigidOptions(
    dt=0.01,
  ),
)
plane=scene.add_entity(gs.morphs.Plane())

os.chdir("../pet/")
robot=scene.add_entity(gs.morphs.MJCF(file="pet.xml"))

B=1024
scene.build(n_envs=B,env_spacing=(1.0,1.0))

min_val=-4
max_val=4
steps_per_control=20
for i in range(1000):
  if i%steps_per_control==0:
    ctrl=torch.rand((B,15),device=gs.device)*(max_val-min_val)+min_val
    robot.control_dofs_position(ctrl)
  scene.step()





