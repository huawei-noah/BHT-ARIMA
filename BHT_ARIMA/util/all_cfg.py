# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

csv = 'perf_Us_3'

cfg_dict = {
    
    'aux_no_0': 
    {
        'dataset': 'aux_no_0',
        'p': 3,
        'd': 1,
        'q': 1,
        'taus': [1533, 8],
        'Rs': [5, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 3,
        'filename': csv
    },
    'aux_smooth':
    {
        'dataset': 'aux_smooth',
        'p': 3,
        'd': 1,
        'q': 1,
        'taus': [319, 8],
        'Rs': [5, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 3,
        'filename': csv
    },
    'aux_raw':
    {
        'dataset': 'aux_raw',
        'p': 3,
        'd': 1,
        'q': 1,
        'taus': [2246, 8],
        'Rs': [5, 8],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 3,
        'filename': csv
    },
    'D1':
    {
        'dataset': 'D1_qty',
        'p': 3,
        'd': 1,
        'q': 1,
        'taus': [607, 10],
        'Rs': [20, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 3,
        'filename': csv
    },
    'PC_W':
    {
        'dataset': 'PC_W',
        'p': 3,
        'd': 1,
        'q': 1,
        'taus': [9, 10],
        'Rs': [5, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 3,
        'filename': csv
    },
    'PC_M':
    {
        'dataset': 'PC_M',
        'p': 3,
        'd': 1,
        'q': 1,
        'taus': [9, 10],
        'Rs': [5, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 3,
        'filename': csv
    },
    'ele40':
    {
        'dataset': 'ele40',
        'p': 3,
        'd': 2,
        'q': 1,
        'taus': [321, 20],
        'Rs': [5, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 3,
        'filename': csv
    },
    'ele200':
    {
        'dataset': 'ele_small',
        'p': 3,
        'd': 2,
        'q': 1,
        'taus': [321, 20],
        'Rs': [5, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 3,
        'filename': csv
    },
    'ele_big':
    {
        'dataset': 'ele_big',
        'p': 3,
        'd': 2,
        'q': 1,
        'taus': [321, 20],
        'Rs': [5, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 1,
        'filename': csv
    },
    'traffic_40':
    {
        'dataset': 'traffic_40',
        'p': 3,
        'd': 2,
        'q': 1,
        'taus': [228, 5],
        'Rs': [20, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 3,
        'filename': csv
    },
    'traffic_80':
    {
        'dataset': 'traffic_small',
        'p': 3,
        'd': 2,
        'q': 1,
        'taus': [228, 5],
        'Rs': [20, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 3,
        'filename': csv
    },
    'traffic_big':
    {
        'dataset': 'traffic_big',
        'p': 3,
        'd': 2,
        'q': 1,
        'taus': [862, 10],
        'Rs': [20, 5],
        'k': 15,
        'tol': 0.001,
        'testsize': 0.1,
        'loop_time': 5,
        'info': 'v2',
        'Us_mode': 1,
        'filename': csv
    }
    
}
