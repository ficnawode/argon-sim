import json
import numpy as np
import os


def json_load(filename: str):
    with open(filename) as f:
        return json.load(f)


def json_dump(d: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(d, filename)


def xyz_write(filename: str, r: np.array):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        f.write(f'{len(r)}\n')
        f.write(f'# comment\n')
        for i in range(len(r)):
            f.write(f'Ar {r[i][0]} {r[i][1]} {r[i][2]}\n')
        f.write(f'\n')


def xyz_append(filename: str, r: np.array):
    with open(filename, 'a') as f:
        f.write(f'{len(r)}\n')
        f.write(f'# comment\n')
        for i in range(len(r)):
            f.write(f'Ar {r[i][0]} {r[i][1]} {r[i][2]}\n')
        f.write(f'\n')
