import json
import numpy as np
import os
from numba import njit, jit
from numba import typed, types


def json_load(filename: str):
    f = open(filename, 'r')
    return json.load(f)


def json_dump(d: dict, filename: str):
    json.dump(d, filename)


def xyz_write(filename: str, r: np.array):
    if os.path.exists(filename):
        os.remove(filename)
    f = open(filename, 'w')
    f.write(f'{len(r)}\n')
    f.write(f'# comment\n')
    for i in range(len(r)):
        f.write(f'Ar {r[i][0]} {r[i][1]} {r[i][2]}\n')
    f.write(f'\n')


def xyz_append(filename: str, r: np.array):
    f = open(filename, 'a')
    f.write(f'{len(r)}\n')
    f.write(f'# comment\n')
    for i in range(len(r)):
        f.write(f'Ar {r[i][0]} {r[i][1]} {r[i][2]}\n')
    f.write(f'\n')
