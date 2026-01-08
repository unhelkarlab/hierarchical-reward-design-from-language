import numpy as np
import rw4t.utils as rw4t_utils

ten_by_ten_0 = np.array([[0, 2, 3, 0, 0, 2, 0, 1, 0, 0],
                         [0, 3, 0, 1, 1, 0, 1, 2, 0, 0],
                         [0, 1, 0, 0, 1, 0, 1, 1, 3, 0],
                         [0, 0, 1, 0, 0, 0, 1, 2, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 3, 0, 1],
                         [1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                         [0, 0, 0, 1, 2, 0, 0, 1, 0, 0],
                         [0, 0, 0, 3, 3, 0, 3, 0, 0, 0],
                         [0, 1, 1, 0, 1, 1, 2, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
ten_by_ten_1 = np.array([[0, 2, 3, 0, 0, 2, 0, 0, 0, 0],
                         [0, 3, 0, 0, 0, 0, 3, 2, 0, 0],
                         [0, 0, 0, 0, 0, 0, 3, 3, 3, 0],
                         [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                         [0, 0, 0, 3, 3, 0, 3, 3, 0, 0],
                         [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
five_by_five_0 = np.array([[2, 0, 3, 0, 0], [0, 3, 0, 2, 0], [0, 0, 0, 0, 3],
                           [0, 3, 3, 0, 0], [0, 0, 0, 0, 2]])
five_by_five_1 = np.array([[0, 2, 3, 0, 0], [0, 3, 0, 0, 2], [0, 0, 0, 0, 3],
                           [0, 3, 3, 0, 0], [0, 0, 2, 0, 0]])

# 6by6 v1, same layout
six_by_six_1_train = np.array([[0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 3, 0],
                               [0, 0, 0, 0, 0, 2], [0, 3, 3, 0, 0, 0],
                               [0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 2]])
# 6by6 v1, different layout
six_by_six_1_diff = np.array([[0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 3, 0],
                              [0, 0, 0, 0, 0, 2], [0, 3, 3, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
# 6by6 v2, same layout
six_by_six_2_train = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 2, 0],
                               [1, 1, 0, 1, 1, 0], [2, 1, 0, 0, 0, 0],
                               [3, 1, 0, 3, 3, 0], [0, 0, 0, 0, 2, 0]])
# 6by6 v2, different layout
six_by_six_2_diff = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 2, 0],
                              [3, 1, 0, 1, 1, 0], [2, 1, 0, 0, 0, 0],
                              [3, 1, 0, 3, 3, 0], [0, 0, 0, 0, 2, 0]])
# 6by6 v3, same layout
six_by_six_3_train = np.array([[0, 0, 0, 2, 0, 0], [1, 3, 0, 0, 0, 2],
                               [1, 0, 0, 0, 0, 1], [0, 0, 0, 3, 0, 1],
                               [3, 0, 0, 0, 0, 0], [2, 3, 0, 0, 2, 0]])
# 6by6 v3, different layout
six_by_six_3_diff = np.array([[0, 0, 0, 1, 0, 0], [1, 3, 0, 0, 0, 2],
                              [1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 1],
                              [3, 0, 0, 0, 0, 0], [2, 3, 0, 0, 2, 0]])
# v4 and v3 have the same layouts
six_by_six_4_train = six_by_six_3_train
six_by_six_4_diff = six_by_six_3_diff

# New configs and maps
six_by_six_5_train_pref_dict = {
    'objects': {
        rw4t_utils.RW4T_State.circle.value: {
            rw4t_utils.RW4T_State.hospital.value: 2
        },
        rw4t_utils.RW4T_State.square.value: {
            rw4t_utils.RW4T_State.school.value: 1
        },
    },
    'zones': [],
    'total_num': 3
}
six_by_six_5_train_map = np.array([[0, 1, 5, 0, 0, 0], [0, 5, 0, 0, 6, 9],
                                   [8, 0, 0, 0, 6, 2], [0, 7, 7, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 6, 3]])

six_by_six_6_train_pref_dict = {
    'objects': {
        rw4t_utils.RW4T_State.circle.value: {
            rw4t_utils.RW4T_State.hospital.value: 2
        },
        rw4t_utils.RW4T_State.square.value: {
            rw4t_utils.RW4T_State.school.value: 1
        },
    },
    'zones': [
        rw4t_utils.RW4T_State.orange_zone.value,
        rw4t_utils.RW4T_State.red_zone.value
    ],
    'total_num':
    3
}

six_by_six_6_train_pref_desc = '''Pick up two red circles and drop them off at the hospital.
Pick up one square and drop it off at the schoo.
Avoid orange and red danger zones.'''

six_by_six_6_train_map = np.array([[0, 1, 5, 0, 0, 0], [0, 5, 0, 0, 6, 9],
                                   [8, 0, 0, 0, 6, 2], [0, 7, 7, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 6, 3]])

ten_by_ten_2_train_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 7, 9, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 7, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 6, 8, 0, 0, 0],
                                   [0, 0, 0, 0, 3, 0, 5, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 5, 6, 3, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

ten_by_ten_2_train_pref_dict = {
    'objects': {
        rw4t_utils.RW4T_State.circle.value: {
            rw4t_utils.RW4T_State.hospital.value: 1
        },
        rw4t_utils.RW4T_State.square.value: {
            rw4t_utils.RW4T_State.school.value: 1
        },
        rw4t_utils.RW4T_State.triangle.value: {
            rw4t_utils.RW4T_State.school.value: 2
        },
    },
    'zones': [
        rw4t_utils.RW4T_State.orange_zone.value,
        rw4t_utils.RW4T_State.red_zone.value
    ],
    'total_num':
    4
}

ten_by_ten_2_train_pref_desc = '''Pick up red circles and drop them off at the hospital.
Pick up squares and drop them off at the school.
Pick up triangles and drop them off at the school.
Avoid orange and red danger zones.'''

ten_by_ten_3_train_map = ten_by_ten_2_train_map

ten_by_ten_3_train_pref_dict = {
    'objects': {
        rw4t_utils.RW4T_State.circle.value: {
            rw4t_utils.RW4T_State.hospital.value: 1
        },
        rw4t_utils.RW4T_State.square.value: {
            rw4t_utils.RW4T_State.school.value: 1
        },
        rw4t_utils.RW4T_State.triangle.value: {
            rw4t_utils.RW4T_State.school.value: 2
        },
    },
    'zones': [],
    'total_num': 4
}

six_by_six_7_train_pref_dict = {
    'objects': {
        rw4t_utils.RW4T_State.circle.value: {
            rw4t_utils.RW4T_State.school.value: 2
        },
        rw4t_utils.RW4T_State.square.value: {
            rw4t_utils.RW4T_State.school.value: 2
        },
    },
    'zones': {
        rw4t_utils.RW4T_HL_Actions_EZ.deliver_circle.value:
        [rw4t_utils.RW4T_State.yellow_zone.value],
        rw4t_utils.RW4T_HL_Actions_EZ.deliver_square.value:
        [rw4t_utils.RW4T_State.yellow_zone.value]
    },
    'total_num': 4
}

six_by_six_8_train_pref_dict = {
    'objects': {
        rw4t_utils.RW4T_State.circle.value: {
            rw4t_utils.RW4T_State.school.value: 2
        },
        rw4t_utils.RW4T_State.square.value: {
            rw4t_utils.RW4T_State.school.value: 2
        },
    },
    'zones': {
        rw4t_utils.RW4T_HL_Actions_EZ.deliver_circle.value:
        [rw4t_utils.RW4T_State.yellow_zone.value],
        rw4t_utils.RW4T_HL_Actions_EZ.deliver_square.value:
        [rw4t_utils.RW4T_State.yellow_zone.value]
    },
    'valid_start_pos': [(0, 0), (0, 5), (5, 5)],
    'total_num': 4
}

six_by_six_7_train_map = np.array(
    [[
        rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
        rw4t_utils.RW4T_State.circle.value, rw4t_utils.RW4T_State.empty.value,
        rw4t_utils.RW4T_State.yellow_zone.value,
        rw4t_utils.RW4T_State.school.value
    ],
     [
         rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value
     ],
     [
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.square.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value
     ],
     [
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.circle.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value
     ],
     [
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.yellow_zone.value
     ],
     [
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.square.value, rw4t_utils.RW4T_State.empty.value
     ]])

six_by_six_8_train_map = np.array(
    [[
        rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
        rw4t_utils.RW4T_State.circle.value, rw4t_utils.RW4T_State.empty.value,
        rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.school.value
    ],
     [
         rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value
     ],
     [
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.square.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value
     ],
     [
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.circle.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.yellow_zone.value
     ],
     [
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.yellow_zone.value
     ],
     [
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.square.value, rw4t_utils.RW4T_State.empty.value
     ]])

pref_dicts = {
    'six_by_six_7_train_pref_dict': six_by_six_7_train_pref_dict,
    'six_by_six_8_train_pref_dict': six_by_six_8_train_pref_dict
}

pref_desc = {
    'six_by_six_6_train_pref_desc': six_by_six_6_train_pref_desc,
    'ten_by_ten_2_train_pref_desc': ten_by_ten_2_train_pref_desc
}

maps = {
    'six_by_six_7_train_map': six_by_six_7_train_map,
    'six_by_six_8_train_map': six_by_six_8_train_map
}
