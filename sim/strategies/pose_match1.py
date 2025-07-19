from sbx import SAC
import numpy as np
from . import pack_observation
from ergon_env2 import ErgonEnv2
import transforms3d
import mujoco

class PoseMatch1:
  """
  Example implementation of a motion imitation strategy using a reference motion.
  """

  episode_len = 1000
  step_number = 0
  algorithm = SAC
  algorithm_config = {'buffer_size': 10000000, 'use_sde': True}
  observation_size = 35
  action_size = 14
  frames_per_action = 20

  home_pose = [0,0,-0.115,1,0.0038399561722115,0,0,
    0,0,-0.881,1.59655,-1.59633,
    0.824582,-1.59655,1.59633,
    0,0,0,0,0,0,0,0]
  home_ctrl = [0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0,0,0]

  prone_pose = [0,-0.197995,-0.174105,0.838151,0.544667,-0.0159519,-0.0242356,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  prone_ctrl = [0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0,0,0]

  supine_pose = [0,0.251216,-0.259594,0.706429,-0.707784,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  supine_ctrl = [0,0,-1.01,2.16,-2.16,0.94,-2.16,2.16,0,0,0,0,0,0,0,0]

  deg2rad = lambda self,deg: deg*np.pi/180.0

  # Terminal conditions parameters
  MAX_TILT_ANGLE = deg2rad(None,45)
  MIN_HEIGHT = -0.2  # meters
  MAX_ACCEL = 50.0  # m/s^2
  POSITION_WEIGHT = 0.5

  def init(self, total_timesteps=0):
    self.env = ErgonEnv2(self)
    self.env.data.qpos[:] = self.home_pose
    self.env.data.qvel[:] = 0
    self.init_qpos = self.env.data.qpos.ravel().copy()
    self.init_qvel = self.env.data.qvel.ravel().copy()
    self.upper_body = np.zeros(8)

    # Define the keyframe script
    self.script = [
      {'time': 10, 'ctrl': [0, 0, -1.01, 2.16, -2.16, 0.94, -2.16, 2.16, 0, 0, 0, 0, 0, 0, 0, 0]},
      {'time': 5, 'ctrl': [0, 0, 0.7, 2.16, 1.57, -0.7, -2.16, -1.57, 0, 0, 0, 0, 0, 0, 0, 0]},
      {'time': 5, 'ctrl': [0, 0, -1.57, 2.16, 1.57, 1.57, -2.16, -1.57, 0, 0, 0, 0, 0, 0, 0, 0]},
      {'time': 5, 'ctrl': [0, 0, -1.57, 0, 1.57, 1.57, 0, -1.57, -1.3, 0, 0, 0, 1.3, 0, 0, 0]},
      {'time': 5, 'ctrl': [0, 0, -1.57, 0, 1.57, 1.57, 0, -1.57, -1.3, 0.8, -1.0, 0, 1.3, -0.8, 1.0, 0]},
      {'time': 5, 'ctrl': [0, 0, -1, 0, 0, 1, 0, 0, 0, 0.0, 0.1, -0.09, 0, -0.0, -0.1, 0.09]},
      {'time': 5, 'ctrl': [0, 0, -1, 1.57, -1.57, 1, -1.57, 1.57, 0, 0.0, 0.1, -0.09, 0, -0.0, -0.1, 0.09]},
    ]

    # Generate the reference trajectory from the keyframe script
    self.generate_reference_trajectory(self.supine_pose, self.script, timestep=1/30.0, interpolation_steps=10)

  def generate_reference_trajectory(self, starting_pose, script, timestep=1/30.0, interpolation_steps=10):
      """
      Generates a reference trajectory by applying the keyframe script to the robot in simulation.

      Parameters:
      - script: List of keyframes, each a dict with 'time' (number of steps) and 'ctrl' (control vector)
      - timestep: Duration of each simulation step (in seconds)
      - interpolation_steps: Number of steps over which to interpolate between keyframes
      """
      print("Generating reference trajectory from keyframes...")
      # Create a separate environment instance for reference generation to avoid interfering with training
      ref_env = ErgonEnv2(self)
      ref_env.reset()
      ref_env.data.qpos[:] = starting_pose
      ref_env.data.qvel[:] = 0.0
      mujoco.mj_step(ref_env.robot,ref_env.data)

      ref_qpos_list = []
      ref_qvel_list = []

      for i, keyframe in enumerate(script):
        current_ctrl = np.array(keyframe['ctrl'][:self.action_size])  # Ensure correct size
        duration = keyframe['time']  # Assuming 'time' is in number of steps
        duration_steps = duration

        if i < len(script) - 1:
          next_ctrl = np.array(script[i+1]['ctrl'][:self.action_size])
          for step in range(duration_steps):
            if step < interpolation_steps:
              alpha = (step + 1) / (interpolation_steps + 1)  # Fraction for interpolation
              interp_ctrl = (1 - alpha) * current_ctrl + alpha * next_ctrl
            else:
              interp_ctrl = current_ctrl
            # Apply interpolated control
            ref_env.data.ctrl[:self.action_size] = interp_ctrl
            # Step the simulation
            mujoco.mj_step(ref_env.robot,ref_env.data)

            # Record the state
            qpos = ref_env.data.qpos.ravel().copy()
            qvel = ref_env.data.qvel.ravel().copy()
            ref_qpos_list.append(qpos)
            ref_qvel_list.append(qvel)
        else:
          # Last keyframe: hold the control for the duration
          for step in range(duration_steps):
            ref_env.data.ctrl[:self.action_size] = current_ctrl
            mujoco.mj_step(ref_env.robot,ref_env.data)

            # Record the state
            qpos = ref_env.data.qpos.ravel().copy()
            qvel = ref_env.data.qvel.ravel().copy()
            ref_qpos_list.append(qpos)
            ref_qvel_list.append(qvel)

      # Convert lists to numpy arrays
      self.ref_qpos = np.array(ref_qpos_list)
      self.ref_qvel = np.array(ref_qvel_list)
      self.ref_length = len(self.ref_qpos)
      print(f"Reference trajectory generated with {self.ref_length} timesteps.")

  def before_sim(self, action):
    self.step_number += 1

  def parametric_gaussian(self, x, mean, sigma):
    coefficient = 1.0/(sigma*np.sqrt(2.0*np.pi))
    exponent = -0.5*((x-mean)/sigma)**2
    return coefficient*np.exp(exponent)

  def quaternion_distance(self, q1, q2):
    # Compute quaternion distance
    dot_product = abs(np.sum(q1 * q2))
    dot_product = min(1.0, max(-1.0, dot_product))  # clamp to [-1,1]
    return 2.0 * np.arccos(dot_product)

  def get_reference_motion(self):
    # Return the reference qpos and qvel for the current timestep.
    # If step_number exceeds ref_length, wrap around or clip.
    idx = min(self.step_number, self.ref_length-1)
    return self.ref_qpos[idx], self.ref_qvel[idx]

  def step(self, action):
    obs = self.observe()

    # Unpack observations
    joint_pos = obs['joint_pos'].copy()  # 21 elements
    chest_accel, torso_accel = obs['accel_data'][:3], obs['accel_data'][3:6]

    # Extract base/root and joint angles
    # Based on notes:
    # qpos[0:3] is base pos, qpos[3:7] base quat, qpos[7:] joints
    base_pos = joint_pos[0:3]
    base_quat = joint_pos[3:7]
    curr_joints = joint_pos[7:]

    curr_qvel = self.env.data.qvel.ravel().copy()
    curr_joint_vel = curr_qvel[6:]  # skip base linear & angular vel, assuming next 15 are joints

    # Get reference pose/velocity
    ref_qpos, ref_qvel = self.get_reference_motion()
    ref_base_pos = ref_qpos[0:3]
    ref_base_quat = ref_qpos[3:7]
    ref_joints = ref_qpos[7:]
    ref_joint_vel = ref_qvel[6:]

    # Compute imitation reward terms:
    # 1. Joint Pose Reward
    joint_pose_error = np.linalg.norm(ref_joints - curr_joints)
    r_pose = np.exp(-5 * joint_pose_error**2)  # weighting as in the paper

    # 2. Joint Velocity Reward
    joint_vel_error = np.linalg.norm(ref_joint_vel - curr_joint_vel)
    r_vel = np.exp(-0.1 * joint_vel_error**2)

    # 3. End-effector (Optional, if you have foot positions)
    # Here we skip due to complexity and unknown foot frame.
    # If desired, you'd get end-effector coords from forward kinematics and compare.
    r_ee = 1.0  # no end-effector tracking implemented here

    # 4. Root Pose Reward
    # Position and orientation matching for the base
    root_pos_error = np.linalg.norm(ref_base_pos - base_pos)
    quat_dist = self.quaternion_distance(ref_base_quat, base_quat)
    r_root_pose = np.exp(-20*root_pos_error**2 -10*quat_dist**2)

    # 5. Root Velocity Reward
    # Similarly, if you have ref velocities for root:
    curr_base_linvel = curr_qvel[0:3]
    curr_base_angvel = curr_qvel[3:6]
    ref_base_linvel = ref_qvel[0:3]
    ref_base_angvel = ref_qvel[3:6]

    linvel_error = np.linalg.norm(ref_base_linvel - curr_base_linvel)
    angvel_error = np.linalg.norm(ref_base_angvel - curr_base_angvel)
    r_root_vel = np.exp(-2*linvel_error**2 -0.2*angvel_error**2)

    # Combine all rewards
    # These weights come from the paper. Adjust as needed.
    w_p = 0.5
    w_v = 0.05
    w_e = 0.2
    w_rp = 0.15
    w_rv = 0.1

    imitation_reward = (w_p * r_pose +
                        w_v * r_vel +
                        w_e * r_ee +
                        w_rp * r_root_pose +
                        w_rv * r_root_vel)

    reward = imitation_reward

    # Check terminal conditions
    terminated = self._check_terminal_conditions(joint_pos, chest_accel, torso_accel)

    # Check truncation
    truncated = self.step_number >= self.ref_length

    return pack_observation(obs), reward, terminated, truncated, {}

  def _check_terminal_conditions(self, joint_pos, chest_accel, torso_accel):
    return False

  def log(self, key, value):
    try:
      self.algorithm_instance.logger.record(key, value)
    except:
      pass

  def reset(self):
    self.step_number = 0
    qpos = np.array(self.supine_pose).ravel().copy()
    qvel = np.array(self.env.data.qvel).ravel().copy()
    self.env.set_state(qpos, qvel)
    return pack_observation(self.observe())

  def observe(self):
    """Get observations including accelerometer data"""
    joint_pos = np.array(self.env.data.qpos)
    chest_accel = self.env.data.sensor('chest').data
    torso_accel = self.env.data.sensor('torso').data
    left_foot_accel = self.env.data.sensor('left_calf').data
    right_foot_accel = self.env.data.sensor('right_calf').data

    # Combine accelerometer data
    accel_data = np.concatenate([chest_accel, torso_accel, left_foot_accel, right_foot_accel])

    return {'joint_pos': joint_pos, 'accel_data': accel_data}

  def apply_action(self, action):
    # Directly apply the action to the actuators
    self.env.data.ctrl[2:] = action
