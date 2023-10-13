import numpy as np
from numba import jit


@jit(nopython=True)
def r(n: int, b_0: np.array, b_1: np.array, b_2: np.array):
    r = []
    for i_0 in range(n):
        for i_1 in range(n):
            for i_2 in range(n):
                r_temp = (i_0 - (n-1)/2)*b_0 + (i_1 - (n-1)/2) * \
                    b_1 + (i_2 - (n-1)/2)*b_2
                r.append(r_temp)
    return r


@jit(nopython=True)
def _E_formula(T_0: float) -> float:
    l = np.random.normal()
    k = 1.380649e-23
    return -1/2*k*T_0*np.log(l)


@jit(nopython=True)
def E(N: int, T_0: float):
    E = []
    for _ in range(N):
        E_temp_0 = _E_formula(T_0)
        E_temp_1 = _E_formula(T_0)
        E_temp_2 = _E_formula(T_0)
        E.append([E_temp_0, E_temp_1, E_temp_2])
    return E


@jit()
def _P_formula(m: float, E_i: float) -> float:
    mult = -1 if np.random.random() < 0.5 else 1
    return mult*np.sqrt(2*m*E_i)


@jit()
def P(N: int, m: float, E: np.array):
    P_0 = []
    P_1 = []
    P_2 = []
    for i in range(N):
        P_0.append(_P_formula(m, E[i][0]))
        P_1.append(_P_formula(m, E[i][1]))
        P_2.append(_P_formula(m, E[i][2]))
    return P_0, P_1, P_2


def b(a: float):
    b_0 = np.array([a, 0, 0])
    b_1 = np.array([a/2, a*np.sqrt(3)/2, 0])
    b_2 = np.array([a/2, a*np.sqrt(3)/2, a*np.sqrt(2/3)])
    return b_0, b_1, b_2
