import sys
import os
import subprocess
import shutil
import numpy as np
from numpy import linalg
from stl import mesh
from math import sqrt
import transforms3d
import pygltflib
import json
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


def deg2rad(degrees):
    return degrees * np.pi / 180


def rad2deg(radians):
    return radians * 180 / np.pi


class Joint:
    def __init__(self, index, type, parent, child, params=None):
        self.index = index
        self.type = type
        self.parent = parent
        self.child = child
        self.params = params

    def __repr__(self):
        return f"Joint(index={self.index},type={self.type},parent={self.parent},child={self.child},params={self.params})"


class Roboscad:

    def translate(self, transform, x, y, z):
        return np.dot(transform, np.array([[1, 0, 0, x],
                                           [0, 1, 0, y],
                                           [0, 0, 1, z],
                                           [0, 0, 0, 1]]))

    def scale(self, transform, x, y, z):
        return np.dot(transform, np.array([[x, 0, 0, 0],
                                           [0, y, 0, 0],
                                           [0, 0, z, 0],
                                           [0, 0, 0, 1]]))

    def xrot(self, transform, angle):
        return np.dot(transform,
                      [[1, 0, 0, 0],
                       [0, np.cos(angle), -np.sin(angle), 0],
                       [0, np.sin(angle), np.cos(angle), 0],
                       [0, 0, 0, 1]])

    def yrot(self, transform, angle):
        return np.dot(transform,
                      [[np.cos(angle), 0, np.sin(angle), 0],
                       [0, 1, 0, 0],
                       [-np.sin(angle), 0, np.cos(angle), 0],
                       [0, 0, 0, 1]])

    def zrot(self, transform, angle):
        return np.dot(transform,
                      [[np.cos(angle), -np.sin(angle), 0, 0],
                       [np.sin(angle), np.cos(angle), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    def rotate(self, transform, x, y, z):
        return self.zrot(self.yrot(self.xrot(transform, x), y), z)

    def mirror(self, transform, x, y, z):
        x_mirror = 1 - x * 2
        y_mirror = 1 - y * 2
        z_mirror = 1 - z * 2
        mirror = [
            [x_mirror, 0, 0, 0],
            [0, y_mirror, 0, 0],
            [0, 0, z_mirror, 0],
            [0, 0, 0, 1]
        ]
        return np.dot(transform, mirror)

    def follow_transform_to_joint(self, joint_id, lines):
        # Start with identity transform
        transform = np.eye(4)
        stack = []
        stack.append(transform)
        for line in lines:
            command = line[0]
            if command == 'joint':
                if line[1] == joint_id:
                    transform = stack[-1]
                    return transform
            elif command == 'translate':
                values = line[1]
                transform = self.translate(transform, values[0], values[1], values[2])
                stack.append(transform)
            elif command == 'rotate':
                values = line[1]
                x = deg2rad(values[0])
                y = deg2rad(values[1])
                z = deg2rad(values[2])
                transform = self.rotate(transform, x, y, z)
                stack.append(transform)
            elif command == 'mirror':
                values = line[1]
                x = values[0]
                y = values[1]
                z = values[2]
                transform = self.mirror(transform, x, y, z)
                stack.append(transform)
            elif command == 'pop':
                stack.pop()
                transform = stack[-1]

    def follow_transform_to_part(self, part, lines):
        # Start with identity transform
        transform = np.eye(4)
        stack = []
        stack.append(transform)
        for line in lines:
            command = line[0]
            values = line[1]
            if command == 'translate':
                transform = self.translate(transform, values[0], values[1], values[2])
                stack.append(transform)
            elif command == 'rotate':
                x = deg2rad(values[0])
                y = deg2rad(values[1])
                z = deg2rad(values[2])
                transform = self.rotate(transform, x, y, z)
                stack.append(transform)
            elif command == 'mirror':
                x = values[0]
                y = values[1]
                z = values[2]
                transform = self.mirror(transform, x, y, z)
                stack.append(transform)
            elif command == 'pop':
                stack.pop()
                transform = stack[-1]
            elif command == 'start_part':
                if values == part:
                    transform = stack[-1]
                    return transform

    def follow_transform_to_sensor(self, sensor_name, lines):
        # Start with identity transform
        transform = np.eye(4)
        stack = []
        stack.append(transform)
        for line in lines:
            command = line[0]
            values = line[1]
            if command == 'translate':
                transform = self.translate(transform, values[0], values[1], values[2])
                stack.append(transform)
            elif command == 'rotate':
                x = deg2rad(values[0])
                y = deg2rad(values[1])
                z = deg2rad(values[2])
                transform = self.rotate(transform, x, y, z)
                stack.append(transform)
            elif command == 'mirror':
                x = values[0]
                y = values[1]
                z = values[2]
                transform = self.mirror(transform, x, y, z)
                stack.append(transform)
            elif command == 'pop':
                stack.pop()
                transform = stack[-1]
            elif command == 'sensor':
                if values == sensor_name:
                    transform = stack[-1]
                    return transform


    def safe_euler_angles(self, roll, pitch, yaw):
        threshold = np.pi / 2 - 1e-3  # Threshold for considering pitch close to ±90°

        if abs(pitch) > threshold:
            # Determine the direction of rotation
            direction = 1 if pitch > 0 else -1

            # Calculate the amount of rotation needed to zero out pitch
            rotation = direction * (np.pi / 2 - abs(pitch))

            # Create rotation matrices
            Ry = np.array([
                [np.cos(rotation), 0, np.sin(rotation)],
                [0, 1, 0],
                [-np.sin(rotation), 0, np.cos(rotation)]
            ])
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])

            # Combine rotations
            R = Rz @ Ry @ Rx

            # Extract new Euler angles
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arcsin(-R[2, 0])
            yaw = np.arctan2(R[1, 0], R[0, 0])

        return roll, pitch, yaw

    def decompose_transform(self, transform, scale_mm=True):
        A = transforms3d.affines.decompose44(np.array(transform))
        xyz = A[0]
        rotation_matrix = A[1]
        scale = A[2]
        roll, pitch, yaw = transforms3d.euler.mat2euler(rotation_matrix, 'rxyz')
        #roll, pitch, yaw = self.safe_euler_angles(roll, pitch, yaw)
        quaternion = transforms3d.quaternions.mat2quat(rotation_matrix)
        if scale_mm:
            # Convert mm to meters
            xyz = (xyz[0] / 1000.0, xyz[1] / 1000.0, xyz[2] / 1000.0)
        return xyz, (roll, pitch, yaw), scale, quaternion

    def get_local_transform(self,transform, parent_transform):
        local_transform = np.matmul(np.linalg.inv(parent_transform), transform)
        # Handle the parent's reflections
        PT = self.decompose_transform(parent_transform, scale_mm=False)
        parent_reflection_matrix = np.eye(4)
        parent_reflection_matrix[0][0] = PT[2][0]
        parent_reflection_matrix[1][1] = PT[2][1]
        parent_reflection_matrix[2][2] = PT[2][2]
        local_transform = np.matmul(parent_reflection_matrix, local_transform)
        return local_transform

    def hex2rgba(self, hex):
        hex = hex.lstrip('#')
        hlen = len(hex)
        if hlen == 3:
            r = int(hex[0] * 2, 16) / 255.0
            g = int(hex[1] * 2, 16) / 255.0
            b = int(hex[2] * 2, 16) / 255.0
            a = 1.0
        elif hlen == 4:
            r = int(hex[0] * 2, 16) / 255.0
            g = int(hex[1] * 2, 16) / 255.0
            b = int(hex[2] * 2, 16) / 255.0
            a = int(hex[3] * 2, 16) / 255.0
        elif hlen == 6:
            r = int(hex[0:2], 16) / 255.0
            g = int(hex[2:4], 16) / 255.0
            b = int(hex[4:6], 16) / 255.0
            a = 1.0
        elif hlen == 8:
            r = int(hex[0:2], 16) / 255.0
            g = int(hex[2:4], 16) / 255.0
            b = int(hex[4:6], 16) / 255.0
            a = int(hex[6:8], 16) / 255.0
        else:
            raise ValueError("Invalid hex string")
        return f'{r} {g} {b} {a}'

    def scad_to_mjcf(self, scad_file, force_update=False):
        scad_file_base_name = os.path.splitext(os.path.basename(scad_file))[0]
        echo_file = f'{scad_file_base_name}.echo'
        mjcf_dir = scad_file_base_name
        meshes_dir = f'{mjcf_dir}/meshes'
        mjcf_file = f'{mjcf_dir}/{scad_file_base_name}.xml'
        robot_name = scad_file_base_name

        # Create the directories, if they don't exist
        if not os.path.exists(mjcf_dir):
            os.makedirs(mjcf_dir)
        if not os.path.exists(meshes_dir):
            os.makedirs(meshes_dir)

        # Process the .scad file
        # openscad -o LowBoy.echo -D roboscad=-1 LowBoy.scad
        process = subprocess.run(['openscad', '-o', echo_file, '-D', 'roboscad=-1', scad_file], check=True)

        # Process the .echo file
        line_number = 0
        all_parts = set()
        parent_parts = {}
        additional_mass = {}
        part_colors = {}
        joints = []
        sensors = []
        part_sensors = {}
        with open(echo_file, 'r') as f:
            lines = []
            for line in f:
                line_number += 1
                if line.startswith('ECHO: "roboscad"'):
                    # example: ECHO: "roboscad", "joint", "bushing", "base", "arm", [90, 270, 0.5, 0.5, 10, 10, 0, 360]
                    without_echo = line.split('ECHO: ')[1]
                    try:
                        data = json.loads(f'[{without_echo}]')
                    except json.JSONDecodeError as e:
                        print(f'Error decoding JSON on line {line_number}: {e}')
                        break
                    command = data[1]
                    # Extract the command
                    if command == 'pop':
                        lines.append((command, None))
                    elif command == 'start_part':
                        part_label = data[2]
                        parent_part_label = data[3]
                        parent_parts[part_label] = parent_part_label
                        additional_mass[part_label] = data[4]
                        part_colors[part_label] = data[5]
                        lines.append((command, part_label, parent_part_label))
                        all_parts.add(part_label)
                    elif command == 'end_part':
                        part_label = data[2]
                        lines.append((command, part_label))
                    elif command == 'joint':
                        type = data[2]
                        parent_part_label = data[3]
                        child_part_label = data[4]
                        values = data[5]
                        joint_id = len(joints)
                        lines.append((command, joint_id, type, parent_part_label, child_part_label, values))
                        joints.append(
                            Joint(joint_id, type, parent_part_label, child_part_label, values))
                    elif command == 'sensor':
                        sensor_name = data[2]
                        sensor_type = data[3]
                        part_label = data[4]
                        size = data[5]
                        lines.append((command, sensor_name, sensor_type, part_label, size))
                        sensors.append((sensor_name, sensor_type, part_label, size))
                        if part_label not in part_sensors:
                            part_sensors[part_label] = []
                        part_sensors[part_label].append(sensor_name)
                    elif len(data) == 3:
                        values = tuple([float(v) for v in data[2]])
                        lines.append((command, values))
                    else:
                        print(f'Error in line({line_number}): {line}')
                        break
        # Create part tree
        part_tree = nx.DiGraph()
        for part in all_parts:
            part_tree.add_node(part)
        for joint in joints:
            part_tree.add_edge(joint.parent, joint.child)

        # Build STLs from the .scad file
        transforms = {}
        cogs = {}
        for part_label in all_parts:
            orig_stl_file = f'{meshes_dir}/part_{part_label}_orig.stl'
            stl_file = f'{meshes_dir}/part_{part_label}.stl'
            if force_update or not os.path.exists(orig_stl_file):
                print(f'Processing part: {part_label}')
                # openscad -o part1.stl -D roboscad=1 LowBoy.scad, hiding output
                process = subprocess.run(
                    ['openscad', '-o', orig_stl_file, '-D', f'roboscad="{part_label}"', '--enable',
                     'all', scad_file],
                    check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            # Load the STL for modification
            stl_mesh = mesh.Mesh.from_file(
                orig_stl_file,
                remove_empty_areas=True,
                remove_duplicate_polygons=True
            )
            
            # Find the transform for the part
            transform = self.follow_transform_to_part(part_label, lines)
            T = self.decompose_transform(transform)

            # Flip the mesh based on the scale component of the transform
            stl_mesh.x = stl_mesh.x * T[2][0]
            stl_mesh.y = stl_mesh.y * T[2][1]
            stl_mesh.z = stl_mesh.z * T[2][2]

            # Scale mm to meters
            stl_mesh.x = stl_mesh.x / 1000
            stl_mesh.y = stl_mesh.y / 1000
            stl_mesh.z = stl_mesh.z / 1000

            # Check if we need to flip normals (if any scale is negative)
            flip_normals = np.prod(T[2]) < 0
            if flip_normals:
                # Reverse the order of vertices in each triangle
                stl_mesh.vectors = stl_mesh.vectors[:, ::-1, :]

            # Update normals and recalculate areas
            stl_mesh.update_normals()
            stl_mesh.update_areas()

            # Calculate the CoG
            pla_density_g_cm3 = 1.25  # g/cm^3
            pla_density = pla_density_g_cm3 * 1000  # kg/m^3
            cog = stl_mesh.get_mass_properties_with_density(pla_density)[2]

            # Offset the STL origin to the CoG
            # stl_mesh.x = stl_mesh.x - cog[0]
            # stl_mesh.y = stl_mesh.y - cog[1]
            # stl_mesh.z = stl_mesh.z - cog[2]

            stl_mesh.save(stl_file)

            # Record the part's transform matrix
            #transform = self.translate(transform, cog[0], cog[1], cog[2])
            transforms[part_label] = transform
            cogs[part_label] = cog
        
        for sensor in sensors:
            sensor_name, sensor_type, _, _ = sensor
            sensor_label = f'sensor_{sensor_name}'
            transform = self.follow_transform_to_sensor(sensor_name, lines)
            transforms[sensor_label] = transform

        # Create the world.xml file
        with open(f'{mjcf_dir}/world.xml', 'w') as f:
            f.write(f"""<mujoco model="World">
  <option timestep="0.005" solver="CG" iterations="30" tolerance="1e-6"/>

  <size memory="20M"/>

  <visual>
    <map force="0.1" zfar="30"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
    <global offwidth="800" offheight="800"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="plane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
             width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="plane" reflectance="0.3" texture="plane" texrepeat="1 1" texuniform="true"/>
  </asset>

  <include file="{robot_name}.xml"/>

  <worldbody>
    <geom name="floor" pos="0 0 -0.3" size="0 0 .25" type="plane" material="plane" condim="3"/>
    <light directional="true" diffuse=".2 .2 .2" specular="0 0 0" pos="0 0 5" dir="0 0 -1" castshadow="false"/>
    <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 4" dir="0 0 -1"/>
  </worldbody>
</mujoco>""")

        # Create the MJCF file
        with open(mjcf_file, 'w') as f:
            f.write(f'<mujoco model="{robot_name}">\n')
            f.write(f"""
  <compiler assetdir="meshes" angle="degree"/>
  <size memory="5M"/>

  <visual>
    <map znear="0.1" zfar="1000 "/>
    <quality shadowsize="8192"/>
  </visual>

  <option timestep="0.005"/>
                      
  <default>
      <default class="servo_joint">
          <joint type="hinge" damping="1.084" armature="0.045" frictionloss="0.03" limited="true"/>
      </default>
      <default class="servo_actuator">
        <position kp="21.1" ctrlrange="-1.570796 1.570796" forcerange="-5 5"/>
      </default>
      <default class="main_vis">
        <geom type="mesh"
              contype="0" conaffinity="0"
              group="0"/>
      </default>
      <default class="main_col">
          <geom type="mesh" solref=".004 1"
           contype="1" conaffinity="1"
           condim="1" friction=".7" 
           group="2"/>
      </default>
  </default>
""")

            f.write(f'  <asset>\n')
            i = 1
            colors=[
                {'name': 'white', 'r': 1, 'g': 1, 'b': 1, 'a': 1.0},
            ] # TODO: Add colors
            for color in colors:
                f.write(f'      <material name="{color['name']}" rgba="{color['r']} {color['g']} {color['b']} {color['a']}"/>\n')
                i += 1

            for part_label in all_parts:
                f.write(f'      <mesh file="part_{part_label}.stl"/>\n')
            f.write(f'  </asset>\n')

            f.write(f'  <worldbody>\n')
                        
            # Walk the part tree
            def walk_tree(part, indent=""):
                part_label = part
                stl_file = f'{meshes_dir}/part_{part_label}.stl'

                transform = transforms[part_label]
                stl_mesh = mesh.Mesh.from_file(stl_file)

                pla_density_g_cm3 = 1.23  # g/cm^3
                pla_density = pla_density_g_cm3 * 1000  # kg/m^3
                volume, vmass, cog, inertia = stl_mesh.get_mass_properties_with_density(pla_density)
                print(inertia)
                print(cog)

                is_root = part_tree.in_degree(part) == 0
                transform = transforms[part]
                
                if is_root or parent_parts[part_label] == "":
                    parent_transform = np.eye(4)
                else:                    
                    parent_part = parent_parts[part_label]
                    parent_transform = transforms[parent_part]

                transform = transforms[part_label]
                local_transform = self.get_local_transform(transform, parent_transform)
                T = self.decompose_transform(local_transform)
                    
                f.write(f'{indent}  <body name="part_{part_label}" ')
                f.write(f'pos="{T[0][0]:.5e} {T[0][1]:.5e} {T[0][2]:.5e}" ')
                f.write(f'quat="{T[3][0]:.5e} {T[3][1]:.5e} {T[3][2]:.5e} {T[3][3]:.5e}" ')
                f.write(f'>\n')
                mass = vmass + additional_mass[part_label] / 1000  # kg
                cog = cogs[part_label]
                f.write(f'{indent}    <inertial pos="{cog[0]:.5e} {cog[1]:.5e} {cog[2]:.5e}" mass="{mass:.5e}" fullinertia="{inertia[0][0]:.5e} {inertia[1][1]:.5e} {inertia[2][2]:.5e} {inertia[0][1]:.5e} {inertia[0][2]:.5e} {inertia[1][2]:.5e}"/>\n')
                # Add any sensor sites
                if part_label in part_sensors:
                    for sensor_name in part_sensors[part_label]:
                        sensor_label = f'sensor_{sensor_name}'
                        sensor_transform = transforms[sensor_label]
                        print(f"{sensor_label} sensor_transform: {self.decompose_transform(sensor_transform)}")
                        print(f"{part_label} transform: {self.decompose_transform(transform)}")
                        local_sensor_transform = self.get_local_transform(sensor_transform, transform)
                        sensor_T = self.decompose_transform(local_sensor_transform)
                        f.write(f'{indent}    <site name="{sensor_name}_site" ')
                        f.write(f'pos="{sensor_T[0][0]:.5e} {sensor_T[0][1]:.5e} {sensor_T[0][2]:.5e}" ')
                        f.write(f'quat="{sensor_T[3][0]:.5e} {sensor_T[3][1]:.5e} {sensor_T[3][2]:.5e} {sensor_T[3][3]:.5e}" ')
                        sensor = [s for s in sensors if s[0] == sensor_name][0]
                        sensor_type = sensor[1]
                        if(sensor_type == 'touch'):
                            print(sensor)
                            size = sensor[3]
                            f.write(f'size="{size[0]/2000.0} {size[1]/2000.0} {size[2]/2000.0}" type="ellipsoid" rgba="0 1 0 1"/>\n')
                        else:
                            f.write(f'size="0.01" type="sphere" rgba="1 0 0 1"/>\n')
                        
                if is_root:
                    f.write(f'{indent}    <freejoint/>\n')

                rgba = self.hex2rgba(part_colors[part_label])
                f.write(f'{indent}    <geom class="main_vis" mesh="part_{part_label}" rgba="{rgba}"/>\n')
                f.write(f'{indent}    <geom class="main_col" mesh="part_{part_label}"/>\n')

                # Find parent joints
                parent_joints = [joint for joint in joints if joint.child == part]
                for joint in parent_joints:
                    f.write(f'{indent}    <joint name="{joint.type}_{joint.parent}_{joint.child}_{joint.params[8].replace(' ','_')}" ')
                    f.write(f'axis="{joint.params[8]}" ')
                    f.write(f'range="{(joint.params[6]):.5e} {(joint.params[7]):.5e}" ')
                    f.write(f'class="servo_joint"/>\n')

                for child in part_tree.successors(part):
                    walk_tree(child, indent + "  ")
                f.write(f'{indent}  </body>\n')

            root_node = [node for node in part_tree.nodes() if part_tree.in_degree(node) == 0][0]
            walk_tree(root_node, "  ")
            f.write(f'  </worldbody>\n')

            # Add sensors
            f.write(f'  <sensor>\n')
            for sensor in sensors:
                sensor_name, sensor_type, _, _ = sensor
                if sensor_type == 'accelerometer':
                    f.write(f'    <accelerometer name="{sensor_name}" site="{sensor_name}_site" />\n')
                    f.write(f'    <gyro name="{sensor_name}_gyro" site="{sensor_name}_site" />\n')
                if sensor_type == 'touch':
                    f.write(f'    <touch name="{sensor_name}" site="{sensor_name}_site" />\n')
            f.write(f'  </sensor>\n')

            f.write(f'  <actuator>\n')
            for joint in joints:
                """ kp="500" kv="10" """
                f.write(f'    <position joint="{joint.type}_{joint.parent}_{joint.child}_{joint.params[8].replace(' ','_')}" class="servo_actuator"/>\n')
            f.write(f'  </actuator>\n')
#             f.write(f"""
#   <keyframe>
#         <key
#         qpos='0 0 -0.115 1 0.0038399561722115 0 0 0 0 -0.881 1.59655 -1.59633 0.824582 -1.59655 1.59633 0 0 0 0 0 0 0 0'
#         ctrl='0 0 -1.01 2.16 -2.16 0.94 -2.16 2.16 0 0 0 0 0 0 0 0'
#         />
#   </keyframe>
# """)
            f.write('</mujoco>\n')

if __name__=="__main__":
    import argparse

    parser=argparse.ArgumentParser(description="Process a .scad file into MJCF format.")
    parser.add_argument('scad_file',type=str,help="Path to the .scad file")
    parser.add_argument('--force-update',action='store_true',help="Force update the output files")

    args=parser.parse_args()
    scad_file=args.scad_file

    if not os.path.exists(scad_file):
        print(f"The file {scad_file} does not exist")
    elif not scad_file.endswith('.scad'):
        print(f"The file {scad_file} is not a .scad file")
    else:
        roboscad_object=Roboscad()
        roboscad_object.scad_to_mjcf(scad_file,force_update=args.force_update)
