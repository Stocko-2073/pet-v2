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

Note: All actuators limited to [-2.156, 2.156] range

<accelerometer name="chest" site="chest_site" />
<accelerometer name="torso" site="torso_site" />
<accelerometer name="left_foot" site="left_foot_site" />
<accelerometer name="right_foot" site="right_foot_site" />
