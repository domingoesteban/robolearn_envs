JOINT_NAMES = [
    'WaistLat',       # Joint 00
    'WaistYaw',       # Joint 01
    'RShSag',         # Joint 02
    'RShLat',         # Joint 03
    'RShYaw',         # Joint 04
    'RElbj',          # Joint 05
    'RForearmPlate',  # Joint 06
    'RWrj1',          # Joint 07
    'RWrj2',          # Joint 08
    'LShSag',         # Joint 09
    'LShLat',         # Joint 10
    'LShYaw',         # Joint 11
    'LElbj',          # Joint 12
    'LForearmPlate',  # Joint 13
    'LWrj1',          # Joint 14
    'LWrj2',          # Joint 15
    'RHipLat',        # Joint 16
    'RHipSag',        # Joint 17
    'RHipYaw',        # Joint 18
    'RKneePitch',     # Joint 19
    'RAnklePitch',    # Joint 20
    'RAnkleRoll',     # Joint 21
    'LHipLat',        # Joint 22
    'LHipSag',        # Joint 23
    'LHipYaw',        # Joint 24
    'LKneePitch',     # Joint 25
    'LAnklePitch',    # Joint 26
    'LAnkleRoll',     # Joint 27
]

INIT_CONFIG = [
   0.0,        # 00 - WaistLat
   0.0,        # 01 - WaistYaw
   0.959931,   # 02 - RShSag
   -0.007266,  # 03 - RShLat
   0.0,        # 04 - RShYaw
   -1.919862,  # 05 - RElbj
   0.0,        # 06 - RForearmPlate
   -0.523599,  # 07 - RWrj1
   0.0,        # 08 - RWrj2
   0.959931,   # 09 - LShSag
   0.007266,   # 10 - LShLat
   0.0,        # 11 - LShYaw
   -1.919862,  # 12 - LElbj
   0.0,        # 13 - LForearmPlate
   -0.523599,  # 14 - LWrj1
   0.0,        # 15 - LWrj2
   0.0,        # 16 - RHipLat
   -0.363826,  # 17 - RHipSag
   0.0,        # 18 - RHipYaw
   0.731245,   # 19 - RKneePitch
   -0.307420,  # 20 - RAnklePitch
   0.0,        # 21 - RAnkleRoll
   0.0,        # 22 - LHipLat
   -0.363826,  # 23 - LHipSag
   0.0,        # 24 - LHipYaw
   0.731245,   # 25 - LKneePitch
   -0.307420,  # 26 - LAnklePitch
   0.0,        # 27 - LAnkleRoll
]

INIT_CONFIG_N_POSE = [
    0.0,        # 00 - WaistLat
    0.0,        # 01 - WaistYaw
    0.0,        # 02 - RShSag
    -0.000,     # 03 - RShLat
    0.0,        # 04 - RShYaw
    0.0,        # 05 - RElbj
    0.0,        # 06 - RForearmPlate
    0.0,        # 07 - RWrj1
    0.0,        # 08 - RWrj2
    0.0,        # 09 - LShSag
    0.000,      # 10 - LShLat
    0.0,        # 11 - LShYaw
    0.0,        # 12 - LElbj
    0.0,        # 13 - LForearmPlate
    0.0,        # 14 - LWrj1
    0.0,        # 15 - LWrj2
    0.0,        # 16 - RHipLat
    0.0,        # 17 - RHipSag
    0.0,        # 18 - RHipYaw
    0.0,        # 19 - RKneePitch
    0.0,        # 20 - RAnklePitch
    0.0,        # 21 - RAnkleRoll
    0.0,        # 22 - LHipLat
    0.0,        # 23 - LHipSag
    0.0,        # 24 - LHipYaw
    0.0,        # 25 - LKneePitch
    0.0,        # 26 - LAnklePitch
    0.0,        # 27 - LAnkleRoll
]

BODY_PARTS = {
    'RA': list(range(2, 9)),
    'LA': list(range(9, 16)),
    'BA': list(range(2, 16)),
    'UB': list(range(0, 16)),
    'LB': list(range(0, 2)) + list(range(16, 28)),
    'BL': list(range(16, 28)),
    'RL': list(range(16, 22)),
    'LL': list(range(22, 28)),
    'RAnkle': list(range(20, 22)),
    'LAnkle': list(range(26, 28)),
    'WB': list(range(0, 28)),
}
