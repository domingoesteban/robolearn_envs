JOINT_NAMES = ['torso_yaw',  # Joint 00

               'j_arm1_1',  # Joint 01 (left arm)
               'j_arm1_2',  # Joint 02
               'j_arm1_3',  # Joint 03
               'j_arm1_4',  # Joint 04
               'j_arm1_5',  # Joint 05
               'j_arm1_6',  # Joint 06
               'j_arm1_7',  # Joint 07

               'j_arm2_1',  # Joint 08 (right arm)
               'j_arm2_2',  # Joint 09
               'j_arm2_3',  # Joint 10
               'j_arm2_4',  # Joint 11
               'j_arm2_5',  # Joint 12
               'j_arm2_6',  # Joint 13
               'j_arm2_7',  # Joint 14

               'hip_yaw_1',  # Joint 15
               'hip_pitch_1',  # Joint 16
               'knee_pitch_1',  # Joint 17

               'hip_yaw_2',  # Joint 18
               'hip_pitch_2',  # Joint 19
               'knee_pitch_2',  # Joint 20

               'hip_yaw_3',  # Joint 21
               'hip_pitch_3',  # Joint 22
               'knee_pitch_3',  # Joint 23

               'hip_yaw_4',  # Joint 24
               'hip_pitch_4',  # Joint 25
               'knee_pitch_4',  # Joint 26

               'ankle_pitch_1',  # Joint 27  #FL
               'ankle_yaw_1',  # Joint 28
               'j_wheel_1',  # Joint 29

               'ankle_pitch_2',  # Joint 30  #FR
               'ankle_yaw_2',  # Joint 31
               'j_wheel_2',  # Joint 32


               'ankle_pitch_3',  # Joint 33  #BL
               'ankle_yaw_3',  # Joint 34
               'j_wheel_3',  # Joint 35

               'ankle_pitch_4',  # Joint 36  #BR
               'ankle_yaw_4',  # Joint 37
               'j_wheel_4',  # Joint 38

               'neck_yaw',  # Joint 39
               'neck_pitch',  # Joint 40
               # 'j_ft_arm2',
               # 'imu_joint',
               ]

# TODO: WE ARE CHANGING ORDER IN LEG JOINTS
INIT_CONFIG = [0.0,  # Joint 00

               0.0,  # Joint 01
               -0.3,  # Joint 02
               -0.8,  # Joint 03
               -0.8,  # Joint 04
               0.0,  # Joint 05
               -0.8,  # Joint 06
               0.0,  # Joint 07

               0.0,  # Joint 08
               0.3,  # Joint 09
               0.8,  # Joint 10
               0.8,  # Joint 11
               0.0,  # Joint 12
               0.8,  # Joint 13
               0.0,  # Joint 14

               0.8,  # Joint 15
               1.0,  # Joint 16
               -1.0,  # Joint 17

               0.8,  # Joint 18
               1.0,  # Joint 19
               -1.0,  # Joint 20

               -0.8,  # Joint 21
               1.0,  # Joint 22
               -1.0,  # Joint 23

               -0.8,  # Joint 24
               1.0,  # Joint 25
               -1.0,  # Joint 26

               0.0,  # Joint 27
               0.8,  # Joint 28
               0.0,  # Joint 29

               0.0,  # Joint 30
               -0.8,  # Joint 31
               0.0,  # Joint 32

               0.0,  # Joint 33
               -0.8,  # Joint 34
               0.0,  # Joint 35

               0.0,  # Joint 36
               0.8,  # Joint 37
               0.0,  # Joint 38

               0.0,  # Joint 39
               0.8,  # Joint 40
               ]

BODY_PARTS = {'LA': list(range(1, 8)),
              'RA': list(range(8, 15)),
              'BA': list(range(1, 15)),
              'TO': [0],
              'HE': list(range(39, 41)),
              'UB': list(range(1, 15))+list(range(39, 41)),
              'FL': list(range(27, 30)),
              'FR': list(range(30, 33)),
              'BL': list(range(33, 36)),
              'BR': list(range(36, 39)),
              'LEGS': list(range(27, 39)),
              'WB': list(range(0, 41))}

TORQUE_LIMITS = [
    (-147.0, 147.0),  # Joint 00
    (-147.0, 147.0),  # Joint 01
    (-147.0, 147.0),  # Joint 02
    (-147.0, 147.0),  # Joint 03
    (-147.0, 147.0),  # Joint 04
    (-55.0, 55.0),  # Joint 05
    (-55.0, 55.0),  # Joint 06
    (-27.0, 27.0),  # Joint 07
    (-147.0, 147.0),  # Joint 08
    (-147.0, 147.0),  # Joint 09
    (-147.0, 147.0),  # Joint 10
    (-147.0, 147.0),  # Joint 11
    (-55.0, 55.0),  # Joint 12
    (-55.0, 55.0),  # Joint 13
    (-27.0, 27.0),  # Joint 14
    (-304.0, 304.0),  # Joint 15
    (-304.0, 304.0),  # Joint 16
    (-304.0, 304.0),  # Joint 17
    (-304.0, 304.0),  # Joint 18
    (-304.0, 304.0),  # Joint 19
    (-304.0, 304.0),  # Joint 20
    (-304.0, 304.0),  # Joint 21
    (-304.0, 304.0),  # Joint 22
    (-304.0, 304.0),  # Joint 23
    (-304.0, 304.0),  # Joint 24
    (-304.0, 304.0),  # Joint 25
    (-304.0, 304.0),  # Joint 26
    (-147.0, 147.0),  # Joint 27
    (-35.0, 35.0),  # Joint 28
    (-35.0, 35.0),  # Joint 29
    (-147.0, 147.0),  # Joint 30
    (-35.0, 35.0),  # Joint 31
    (-35.0, 35.0),  # Joint 32
    (-147.0, 147.0),  # Joint 33
    (-35.0, 35.0),  # Joint 34
    (-35.0, 35.0),  # Joint 35
    (-147.0, 147.0),  # Joint 36
    (-35.0, 35.0),  # Joint 37
    (-35.0, 35.0),  # Joint 38
    (-35.0, 35.0),  # Joint 39
    (-35.0, 35.0),  # Joint 40
                 ]

POSITION_LIMITS = [
    (-2.618, 2.618),  # Joint 00
    (-3.403, 1.658),  # Joint 01
    (-3.49, 0.0),  # Joint 02
    (-2.618, 2.618),  # Joint 03
    (-2.53, 0.349),  # Joint 04
    (-2.618, 2.618),  # Joint 05
    (-1.57, 1.57),  # Joint 06
    (-2.618, 2.618),  # Joint 07
    (-1.658, 3.403),  # Joint 08
    (0.0, 3.49),  # Joint 09
    (-2.618, 2.618),  # Joint 10
    (-0.349, 2.53),  # Joint 11
    (-2.618, 2.618),  # Joint 12
    (-1.57, 1.57),  # Joint 13
    (-2.618, 2.618),  # Joint 14
    (-1.13446401, 2.70526034),  # Joint 15
    (-2.0943951, 2.0943951),  # Joint 16
    (-2.61799388, 2.61799388),  # Joint 17
    (-2.70526034, 1.13446401),  # Joint 18
    (-2.0943951, 2.0943951),  # Joint 19
    (-2.61799388, 2.61799388),  # Joint 20
    (-2.70526034, 1.13446401),  # Joint 21
    (-2.0943951, 2.0943951),  # Joint 22
    (-2.61799388, 2.61799388),  # Joint 23
    (-1.13446401, 2.70526034),  # Joint 24
    (-2.0943951, 2.0943951),  # Joint 25
    (-2.61799388, 2.61799388),  # Joint 26
    (-2.61799388, 2.61799388),  # Joint 27
    (-2.58308729, 2.58308729),  # Joint 28
    (0.0, -1.0),  # Joint 29
    (-2.61799388, 2.61799388),  # Joint 30
    (-2.58308729, 2.58308729),  # Joint 31
    (0.0, -1.0),  # Joint 32
    (-2.61799388, 2.61799388),  # Joint 33
    (-2.58308729, 2.58308729),  # Joint 34
    (0.0, -1.0),  # Joint 35
    (-2.61799388, 2.61799388),  # Joint 36
    (-2.58308729, 2.58308729),  # Joint 37
    (0.0, -1.0),  # Joint 38
    (-0.78539816, 0.78539816),  # Joint 39
    (-0.34906585, 0.78539816),  # Joint 40
    ]

VELOCITY_LIMITS = [
    (-5.7, 5.7),  # Joint 00
    (-5.7, 5.7),  # Joint 01
    (-5.7, 5.7),  # Joint 02
    (-8.2, 8.2),  # Joint 03
    (-8.2, 8.2),  # Joint 04
    (-11.6, 11.6),  # Joint 05
    (-11.6, 11.6),  # Joint 06
    (-20.3, 20.3),  # Joint 07
    (-5.7, 5.7),  # Joint 08
    (-5.7, 5.7),  # Joint 09
    (-8.2, 8.2),  # Joint 10
    (-8.2, 8.2),  # Joint 11
    (-11.6, 11.6),  # Joint 12
    (-11.6, 11.6),  # Joint 13
    (-20.3, 20.3),  # Joint 14
    (-8.8, 8.8),  # Joint 15
    (-8.8, 8.8),  # Joint 16
    (-8.8, 8.8),  # Joint 17
    (-8.8, 8.8),  # Joint 18
    (-8.8, 8.8),  # Joint 19
    (-8.8, 8.8),  # Joint 20
    (-8.8, 8.8),  # Joint 21
    (-8.8, 8.8),  # Joint 22
    (-8.8, 8.8),  # Joint 23
    (-8.8, 8.8),  # Joint 24
    (-8.8, 8.8),  # Joint 25
    (-8.8, 8.8),  # Joint 26
    (-8.1, 8.1),  # Joint 27
    (-20.0, 20.0),  # Joint 28
    (-20.0, 20.0),  # Joint 29
    (-8.1, 8.1),  # Joint 30
    (-20.0, 20.0),  # Joint 31
    (-20.0, 20.0),  # Joint 32
    (-8.1, 8.1),  # Joint 33
    (-20.0, 20.0),  # Joint 34
    (-20.0, 20.0),  # Joint 35
    (-8.1, 8.1),  # Joint 36
    (-20.0, 20.0),  # Joint 37
    (-20.0, 20.0),  # Joint 38
    (-5.7, 5.7),  # Joint 39
    (-5.7, 5.7),  # Joint 40
                  ]
