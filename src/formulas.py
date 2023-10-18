import numpy as np
from numba import jit


from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

# (4)


def b(a: float):
    b_0 = np.array([a, 0, 0])
    b_1 = np.array([a/2, a*np.sqrt(3)/2, 0])
    b_2 = np.array([a/2, a*np.sqrt(3)/6, a*np.sqrt(2/3)])
    return b_0, b_1, b_2


# (5)
@jit(nopython=True)
def r(n: int, b_0: np.array, b_1: np.array, b_2: np.array):
    def _r_formula(i_0, i_1, i_2, b_0, b_1, b_2):
        return (i_0 - (n-1)/2)*b_0 + (i_1 - (n-1)/2) * b_1 + (i_2 - (n-1)/2)*b_2

    r = []
    for i_0 in range(n):
        for i_1 in range(n):
            for i_2 in range(n):
                r_temp = _r_formula(i_0, i_1, i_2, b_0, b_1, b_2)
                r.append(r_temp)
    return r


# (6)
@jit(nopython=True)
def E(N: int, T_0: float):

    def _E_formula(T_0: float) -> float:
        l = np.random.random()
        k = 8.31e-3
        return -1/2*k*T_0*np.log(l)

    return [[_E_formula(T_0), _E_formula(T_0), _E_formula(T_0)] for _ in range(N)]


# (7)
@jit
def P(N: int, m: float, E: np.array):

    def _P_formula(m: float, E_i: float) -> float:
        mult = -1 if np.random.random() < 0.5 else 1
        return mult*np.sqrt(2*m*E_i)

    return [[_P_formula(m, E[i][0]), _P_formula(m, E[i][1]), _P_formula(m, E[i][2])] for i in range(N)]


# (8)
# @jit(nopython=True)
# def P_prime(P: np.array) -> np.array:
#     P_sum = np.sum(P)/len(P)
#     return [p - P_sum for p in P]


# (9)
@jit
def V_P(r_i: np.array, r_j: np.array, epsilon: float, R: float) -> float:
    r_ij = np.linalg.norm(r_i - r_j)
    return epsilon*((R/r_ij)**12 - 2*(R/r_ij)**6)


# (10)
@jit
def V_S(r_i: float, L: float, f: float) -> float:
    r_i = np.linalg.norm(r_i)
    if r_i < L:
        return 0
    return f/2*(r_i - L)**2


# (11)
@jit
def V(r: np.array, epsilon: float, R: float, L: float, f: float) -> float:
    N = len(r)

    V_P_sum = 0
    for i in range(N):
        for j in range(i):
            if i == j:
                continue
            V_P_sum += V_P(r[i], r[j], epsilon, R)

    V_S_sum = 0
    for i in range(N):
        V_S_sum += V_S(r[i], L, f)

    return V_P_sum + V_S_sum


# (13)
@jit
def F_P(r, epsilon: float, R: float) -> np.array:
    N = len(r)

    def _F_P_formula(r_i, r_j, epsilon, R):
        r_ij = np.linalg.norm(r_i - r_j)
        return 12*epsilon*((R/r_ij)**12 - (R/r_ij)**6)*(r_i - r_j)/r_ij**2

    def _F_P_sum(r, i, epsilon, R):
        F_P_sum = 0
        for j in range(N):
            if i == j:
                continue
            F_P_sum += _F_P_formula(r[i], r[j], epsilon, R)
        return F_P_sum

    return np.array([_F_P_sum(r, i, epsilon, R) for i in range(N)])


# (14)
@jit
def F_S(r: np.array, L: float, f: float) -> np.array:
    def _F_S_formula(r_i, L, f):
        r_i_norm = np.linalg.norm(r_i)
        if r_i_norm < L:
            return [0, 0, 0]
        return np.array(f*(L-r_i_norm)*r_i/r_i_norm)
    return np.array([_F_S_formula(r_i, L, f) for r_i in r])


# (12)
@jit(nopython=True)
def F(r, epsilon, R, L, f) -> np.array:
    return F_P(r, epsilon, R) + F_S(r, L, f)


# (15)
@jit
def p(F_S: np.array, L: float) -> np.array:
    F_S_x = [f[0] for f in F_S]
    F_S_y = [f[1] for f in F_S]
    F_S_z = [f[2] for f in F_S]
    F_S_sum = np.array([np.sum(F_S_x), np.sum(F_S_y), np.sum(F_S_z)])
    return 1/(4*np.pi*L**2)*np.linalg.norm(F_S_sum)


# (16)
@jit(nopython=True)
def H(p: np.array, m: float, V: float):
    '''
    p is momentum, m is mass, V is potential
    '''
    return np.sum([np.linalg.norm(p_i)**2/(2*m) for p_i in p]) + V


# (18a+c)
@jit
def P_advance(p: np.array, F: np.array, tau: float):
    return p + 1/2 * F * tau


# (18b)
@jit
def r_advance(r, m, p, tau):
    return r + 1/m * p * tau


# (19)
@jit
def T(p, m):
    k = 8.31e-3
    N = len(p)
    return 2/3 * N/k * np.sum([np.linalg.norm(p_i)**2/(2*m) for p_i in p])
