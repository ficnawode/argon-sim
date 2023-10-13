import sys
import argparse
from typing import List

from matplotlib import pyplot as plt

import formulas
from file_utils import json_load, xyz_dump


class ArgonSimulation():
    def __init__(self, params_filepath: str, xyz_filepath: str, out_filepath: str, show_plots):
        # param keys: n, m, e, R, f, L, a, T_0, tau, S_o, S_d, S_out, S_xyz
        self.params = json_load(params_filepath)
        self.xyz_filepath = xyz_filepath
        self.out_filepath = out_filepath
        self.show_plots = show_plots

    def run(self):
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

            plt.title('P')
            plt.plot(E)
            plt.show()

            plt.title('P')
            plt.plot(P)
            plt.show()

            plt.title('P')
            plt.hist(P, 20)
            plt.show()

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
        nargs=1,
        required=True,
        type=str,
        help="Filename to save output XYZ coordinates into",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        nargs=1,
        required=False,
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
