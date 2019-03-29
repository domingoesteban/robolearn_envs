JOINT_NAMES = [
    'WaistLat',       # 00
    'WaistSag',       # 01
    'WaistYaw',       # 02
    'RShSag',         # 03
    'RShLat',         # 04
    'RShYaw',         # 05
    'RElbj',          # 06
    'RForearmPlate',  # 07
    'RWrj1',          # 08
    'RWrj2',          # 09
    'LShSag',         # 10
    'LShLat',         # 11
    'LShYaw',         # 12
    'LElbj',          # 13
    'LForearmPlate',  # 14
    'LWrj1',          # 15
    'LWrj2',          # 16
    'RHipSag',        # 17
    'RHipLat',        # 18
    'RHipYaw',        # 19
    'RKneeSag',       # 20
    'RAnkLat',        # 21
    'RAnkSag',        # 22
    'LHipSag',        # 23
    'LHipLat',        # 24
    'LHipYaw',        # 25
    'LKneeSag',       # 26
    'LAnkLat',        # 27
    'LAnkSag',        # 28
]

INIT_CONFIG = [
    0.0,  # 00
    0.0,  # 01
    0.0,  # 02
    0.0,  # 03
    0.0,  # 04
    0.0,  # 05
    0.0,  # 06
    0.0,  # 07
    0.0,  # 08
    0.0,  # 09
    0.0,  # 10
    0.0,  # 11
    0.0,  # 12
    0.0,  # 13
    0.0,  # 14
    0.0,  # 15
    0.0,  # 16
    0.0,  # 17
    0.0,  # 18
    0.0,  # 19
    0.0,  # 20
    0.0,  # 21
    0.0,  # 22
    0.0,  # 23
    0.0,  # 24
    0.0,  # 25
    0.0,  # 26
    0.0,  # 27
    0.0,  # 28
]

INIT_CONFIG_N_POSE = [
    0.0,  # 00
    0.0,  # 01
    0.0,  # 02
    0.0,  # 03
    0.0,  # 04
    0.0,  # 05
    0.0,  # 06
    0.0,  # 07
    0.0,  # 08
    0.0,  # 09
    0.0,  # 10
    0.0,  # 11
    0.0,  # 12
    0.0,  # 13
    0.0,  # 14
    0.0,  # 15
    0.0,  # 16
    0.0,  # 17
    0.0,  # 18
    0.0,  # 19
    0.0,  # 20
    0.0,  # 21
    0.0,  # 22
    0.0,  # 23
    0.0,  # 24
    0.0,  # 25
    0.0,  # 26
    0.0,  # 27
    0.0,  # 28
]

BODY_PARTS = {
    # 'RA': list(range(3, 10)),
    # 'LA': list(range(10, 17)),
    # 'UB': list(range(0, 17)),
    'LB': list(range(17, 29)),
    'BL': list(range(17, 29)),
    'RL': list(range(17, 23)),
    'LL': list(range(23, 29)),
    'WB': list(range(0, 3)) + list(range(17, 29)),
}
