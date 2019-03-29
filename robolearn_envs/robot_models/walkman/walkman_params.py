JOINT_NAMES = [
    'LHipLat',        # Joint 00
    'LHipYaw',        # Joint 01
    'LHipSag',        # Joint 02
    'LKneeSag',       # Joint 03
    'LAnkSag',        # Joint 04
    'LAnkLat',        # Joint 05
    'RHipLat',        # Joint 06
    'RHipYaw',        # Joint 07
    'RHipSag',        # Joint 08
    'RKneeSag',       # Joint 09
    'RAnkSag',        # Joint 10
    'RAnkLat',        # Joint 11
    'WaistLat',       # Joint 12
    'WaistSag',       # Joint 13
    'WaistYaw',       # Joint 14
    'LShSag',         # Joint 15
    'LShLat',         # Joint 16
    'LShYaw',         # Joint 17
    'LElbj',          # Joint 18
    'LForearmPlate',  # Joint 19
    'LWrj1',          # Joint 20
    'LWrj2',          # Joint 21
    'NeckYawj',       # Joint 22
    'NeckPitchj',     # Joint 23
    'RShSag',         # Joint 24
    'RShLat',         # Joint 25
    'RShYaw',         # Joint 26
    'RElbj',          # Joint 27
    'RForearmPlate',  # Joint 28
    'RWrj1',          # Joint 29
    'RWrj2',          # Joint 30
]

INIT_CONFIG = [
    -0.05,  # Joint 0
    0.0,    # Joint 1
    -0.50,  # Joint 2
    1.0,    # Joint 3
    -0.45,  # Joint 4
    0.05,   # Joint 5
    0.05,   # Joint 6
    0.0,    # Joint 7
    -0.50,  # Joint 8
    1.0,    # Joint 9
    -0.45,  # Joint 10
    -0.05,  # Joint 11
    0.0,    # Joint 12
    0.1,    # Joint 13
    0.0,    # Joint 14
    0.5,    # Joint 15
    0.25,   # Joint 16
    0.0,    # Joint 17
    -1.0,   # Joint 18
    0.0,    # Joint 19
    0.0,    # Joint 20
    0.0,    # Joint 21
    0.0,    # Joint 22
    0.0,    # Joint 23
    0.5,    # Joint 24
    -0.25,  # Joint 25
    0.0,    # Joint 26
    -1.0,   # Joint 27
    0.0,    # Joint 28
    0.0,    # Joint 29
    0.0,    # Joint 30
]

BODY_PARTS = {
    'RA': list(range(24, 31)),
    'LA': list(range(15, 22)),
    'BA': list(range(24, 31)) + list(range(15, 22)),
    'UB': list(range(24, 31)) + list(range(15, 22)),
    'BL': list(range(15)),
    'RL': list(range(6, 12)),
    'LL': list(range(0, 6)),
    'WB': list(range(0, 31)),
}
