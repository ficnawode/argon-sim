import sys
import argparse
from typing import List

from matplotlib import pyplot as plt
from progress.bar import Bar
import tqdm

import formulas
from file_utils import json_load, xyz_dump, xyz_append


class ArgonSimulation():
    def __init__(self, params_filepath: str, xyz_filepath: str, out_filepath: str, show_plots):
        # param keys: n, m, e, R, f, L, a, T_0, tau, S_o, S_d, S_out, S_xyz
        self.params = json_load(params_filepath)
        self.xyz_filepath = xyz_filepath
        self.out_filepath = out_filepath
        self.show_plots = show_plots

    def run(self):
        with open(self.out_filepath, 'w') as fp:
            fp.write("s, T, E, p\n")
        # initial conditions
        N = self.params['n']**3
        b_0, b_1, b_2 = formulas.b(self.params['a'])
        r = formulas.r(self.params['n'], b_0, b_1, b_2)
        E = formulas.E(N, self.params['T_0'])
        P = formulas.P(N, self.params['m'], E)

        xyz_dump('out.xyz', r)

        if self.show_plots:
            plt.title('r')
            plt.plot(r)
            plt.show()

            plt.title('E')
            plt.plot(E)
            plt.show()

            plt.title('P')
            plt.plot(P)
            plt.show()

            plt.title('P')
            plt.hist([p[0] for p in P], 20)
            plt.hist([p[1] for p in P], 20)
            plt.hist([p[2] for p in P], 20)
            plt.show()

        V = formulas.V(
            r, self.params['e'], self.params['R'], self.params['L'], self.params['f'])
        print(f'\n\n\n V = {V}\n\n')
        F_S = formulas.F_S(r, self.params['L'], self.params['f'])
        F_P = formulas.F_P(r, self.params['e'], self.params['R'])
        F = F_S + F_P
        p = formulas.p(F_S, self.params['L'])

        if self.show_plots:

            plt.title('F_S')
            plt.hist([f[0] for f in F_S], 20)
            plt.hist([f[1] for f in F_S], 20)
            plt.hist([f[2] for f in F_S], 20)
            plt.show()

            plt.title('F_P')
            plt.hist([f[0] for f in F_P], 20)
            plt.hist([f[1] for f in F_P], 20)
            plt.hist([f[2] for f in F_P], 20)
            plt.show()

        T_bar = 0
        P_bar = 0
        H_bar = 0
        for s in tqdm.tqdm(range(self.params['S_o'] + self.params['S_d'])):
            # modify momenta (18a)
            P = formulas.P_advance(P, F, self.params['tau'])
            # modify displacement (18b)
            r = formulas.r_advance(
                r, self.params['m'], p, self.params['tau'])

            # calculate V, F, p
            V = formulas.V(
                r, self.params['e'], self.params['R'], self.params['L'], self.params['f'])
            F_S = formulas.F_S(r, self.params['L'], self.params['f'])
            F_P = formulas.F_P(r, self.params['e'], self.params['R'])
            F = F_S + F_P

            # modify momenta (18c)
            P = formulas.P_advance(P, F, self.params['tau'])

            # calculate T(19), E (16), P()
            T = formulas.T(P, self.params['m'])
            V = formulas.V(
                r, self.params['e'], self.params['R'], self.params['L'], self.params['f'])
            p = formulas.p(F_S, self.params['L'])

            if s % self.params['S_out'] == 0:
                # save current values to file
                with open(self.out_filepath, 'a') as fp:
                    fp.write(f"{s}, {T}, {V}, {p}\n")
                pass
            if s % self.params['S_xyz'] == 0:
                xyz_append(self.xyz_filepath, r)
                # save coordinates to xyz file
                pass
            if s > self.params['S_o'] == 0:
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
