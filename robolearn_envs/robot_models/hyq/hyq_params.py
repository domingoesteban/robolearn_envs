JOINT_NAMES = [
    'lf_haa_joint',       # 00
    'lf_hfe_joint',       # 01
    'lf_kfe_joint',       # 02
    'lh_haa_joint',       # 03
    'lh_hfe_joint',       # 04
    'lh_kfe_joint',       # 05
    'rf_haa_joint',       # 06
    'rf_hfe_joint',       # 07
    'rf_kfe_joint',       # 08
    'rh_haa_joint',       # 09
    'rh_hfe_joint',       # 10
    'rh_kfe_joint',       # 11
]

INIT_CONFIG = [
    0.0,    # 00 lf_haa_joint
    0.0,    # 01 lf_hfe_joint
    -0.35,  # 02 lf_kfe_joint
    0.0,    # 03 lh_haa_joint
    0.0,    # 04 lh_hfe_joint
    0.35,   # 05 lh_kfe_joint
    0.0,    # 06 rf_haa_joint
    0.0,    # 07 rf_hfe_joint
    -0.35,  # 08 rf_kfe_joint
    0.0,    # 09 rh_haa_joint
    0.0,    # 10 rh_hfe_joint
    0.35,   # 11 rh_kfe_joint
]

INIT_CONFIG_N_POSE = [
    0.0,  # 00 lf_haa_joint
    0.0,  # 01 lf_hfe_joint
    0.0,  # 02 lf_kfe_joint
    0.0,  # 03 lh_haa_joint
    0.0,  # 04 lh_hfe_joint
    0.0,  # 05 lh_kfe_joint
    0.0,  # 06 rf_haa_joint
    0.0,  # 07 rf_hfe_joint
    0.0,  # 08 rf_kfe_joint
    0.0,  # 09 rh_haa_joint
    0.0,  # 10 rh_hfe_joint
    0.0,  # 11 rh_kfe_joint
]


INIT_CONFIG_HOMING = [
    -0.2,   # 00 lf_haa_joint
    0.75,   # 01 lf_hfe_joint
    -1.5,   # 02 lf_kfe_joint
    -0.2,   # 03 lh_haa_joint
    -0.75,  # 04 lh_hfe_joint
    1.5,    # 05 lh_kfe_joint
    -0.2,   # 06 rf_haa_joint
    0.75,   # 07 rf_hfe_joint
    -1.5,   # 08 rf_kfe_joint
    -0.2,   # 09 rh_haa_joint
    -0.75,  # 10 rh_hfe_joint
    1.5,    # 11 rh_kfe_joint
]

# hip_roll = 0.2
# hip_pitch = 0.75 # 0.75
# knee_pitch = 1.5 # 1.5
#
# self.home_config = [-hip_roll, hip_pitch, -knee_pitch] + [-hip_roll, -hip_pitch, knee_pitch] + [-hip_roll, hip_pitch, -knee_pitch] + [-hip_roll, -hip_pitch, knee_pitch]




BODY_PARTS = {
    'LL': list(range(0, 3)),
    'LA': list(range(3, 6)),
    'RL': list(range(6, 9)),
    'RA': list(range(9, 12)),
    'WB': list(range(0, 12)),
    'BL': list(range(0, 3)) + list(range(6, 9)),
    'BA': list(range(3, 6)) + list(range(9, 12)),
}

