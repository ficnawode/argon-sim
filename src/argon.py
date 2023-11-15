import sys
import argparse
from typing import List

from matplotlib import pyplot as plt
from progress.bar import Bar
import tqdm
import numba
from numba.experimental import jitclass

import formulas
from file_utils import json_load, xyz_write, xyz_append, json_dump
from numba import njit

config_spec = [
    ("n", numba.int32),
    ("m", numba.float64),
    ("e", numba.float64),
    ("R", numba.float64),
    ("f", numba.float64),
    ("L", numba.float64),
    ("a", numba.float64),
    ("T_0", numba.float64),
    ("tau", numba.float64),
    ("S_o", numba.int32),
    ("S_d", numba.int32),
    ("S_out", numba.int32),
    ("S_xyz", numba.int32),
]


@jitclass(config_spec)
class SimConfig(object):
    def __init__(self, d):
        self.n = d['n']
        self.m = d['m']
        self.e = d['e']
        self.R = d['R']
        self.f = d['f']
        self.L = 1.23*d['a']*(d['n']-1)
        self.a = d['a']
        self.T_0 = d['T_0']
        self.tau = d['tau']
        self.S_o = d['S_o']
        self.S_d = d['S_d']
        self.S_out = d['S_out']
        self.S_xyz = d['S_xyz']


class ArgonSimulation():
    def __init__(self, params_filepath: str, xyz_filepath: str, out_filepath: str, show_plots):
        # param keys: n, m, e, R, f, L, a, T_0, tau, S_o, S_d, S_out, S_xyz
        typed_config = numba.typed.Dict.empty(
            key_type=numba.types.unicode_type,
            value_type=numba.types.float64,
        )
        json_config = json_load(params_filepath)
        for key in json_config:
            typed_config[key] = json_config[key]
        self.params = SimConfig(typed_config)
        self.xyz_filepath = xyz_filepath
        self.out_filepath = out_filepath
        self.show_plots = show_plots

    # @njit
    def run(self):

        with open(self.out_filepath, 'w') as fp:
            fp.write("s, T, E, p, H\n")
        # initial conditions
        N = self.params.n**3
        b_0, b_1, b_2 = formulas.b(self.params.a)
        r = formulas.r(self.params.n, b_0, b_1, b_2)
        # print(r)
        E = formulas.E(N, self.params.T_0)
        P = formulas.P(N, self.params.m, E)

        xyz_write('out.xyz', r)

        V = formulas.V(
            r, self.params.e, self.params.R, self.params.L, self.params.f)
        print(f'\n\n\n V = {V}\n\n')
        F_S = formulas.F_S(r, self.params.L, self.params.f)
        F_P = formulas.F_P(r, self.params.e, self.params.R)
        F = F_S + F_P
        p = formulas.p(F_S, self.params.L)

        T_bar = 0
        P_bar = 0
        H_bar = 0
        for s in tqdm.tqdm(range(self.params.S_o + self.params.S_d)):
            # modify momenta (18a)
            P = formulas.P_advance(P, F, self.params.tau)
            # modify displacement (18b)
            r = formulas.r_advance(
                r, self.params.m, P, self.params.tau)

            # calculate V, F, p
            V = formulas.V(
                r, self.params.e, self.params.R, self.params.L, self.params.f)
            F_S = formulas.F_S(r, self.params.L, self.params.f)
            F_P = formulas.F_P(r, self.params.e, self.params.R)
            F = F_S + F_P

            # modify momenta (18c)
            P = formulas.P_advance(P, F, self.params.tau)

            # calculate T(19), E (16), P()
            T = formulas.T(P, self.params.m)
            V = formulas.V(
                r, self.params.e, self.params.R, self.params.L, self.params.f)
            p = formulas.p(F_S, self.params.L)
            H = formulas.H(P, self.params.m, V)

            if s % self.params.S_out == 0:
                # save current values to file
                with open(self.out_filepath, 'a') as fp:
                    fp.write(f"{s}, {T}, {V}, {p}, {H}\n")
                pass
            if s % self.params.S_xyz == 0:
                xyz_append(self.xyz_filepath, r)
                # save coordinates to xyz file
                pass
            if s > self.params.S_o == 0:
                # accumulate values T P H
                pass

        # save to file

    def __repr__(self):
        return 'ArgonSimulation object: ' + str(self.__dict__)


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser(
        prog="Argon simulation", description="Argon simulation for KMS"
    )
    parser.add_argument(
        "--params",
        "-p",
        required=True,
        type=str,
        help="Filename of path of json parameter file.",
    )
    parser.add_argument(
        "--cordfile",
        "-cf",
        required=True,
        type=str,
        help="Filename to save output XYZ coordinates into",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        required=True,
        type=str,
        help="Path to output file. If not specified, output will be printed to stdout.",
    )
    parser.add_argument(
        "--showplots",
        action="store_true",
        help="Creates plots and prints them without saving to file. Will halt program function.",
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    sim = ArgonSimulation(args.params, args.cordfile,
                          args.outfile, args.showplots)
    sim.run()
