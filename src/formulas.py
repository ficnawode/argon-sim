import numpy as np
from numpy.linalg import norm
from numba import jit, njit


from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


@njit('float64(float64,float64)')
def L_formula(a, n):
    return 1.23*a*(n-1)


@njit('UniTuple(float64[:], 3)(float64)')
def b(a: float):
    b_0 = np.array([a, 0, 0])
    b_1 = np.array([a/2, a*np.sqrt(3)/2, 0])
    b_2 = np.array([a/2, a*np.sqrt(3)/6, a*np.sqrt(2/3)])
    return b_0, b_1, b_2


@njit('float64[:](int32, int32, int32, int32, float64[:], float64[:], float64[:])')
def _r_formula(n, i_0, i_1, i_2, b_0, b_1, b_2):
    r_temp = (i_0 - (n-1)/2)*b_0 + (i_1 - (n-1)/2) * b_1 + (i_2 - (n-1)/2)*b_2
    return r_temp


@njit('float64[:,:](int32, float64[:], float64[:], float64[:])')
def r(n: int, b_0: np.array, b_1: np.array, b_2: np.array):
    r_array = np.zeros((n*n*n, 3), dtype=np.float64)

    for i_0 in range(n):
        for i_1 in range(n):
            for i_2 in range(n):
                index = i_0*n*n+i_1*n + i_2
                r_array[index] = _r_formula(n, i_0, i_1, i_2, b_0, b_1, b_2)
    return r_array


@njit('float64(float64)')
def _E_formula(T_0: float) -> float:
    l = np.random.random()
    k = 8.31e-3
    return -1/2*k*T_0*np.log(l)


# (6)
@njit('float64[:,:](int32, float64)')
def E(N: int, T_0: float):
    E_array = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        l = np.array(
            [np.random.random(), np.random.random(), np.random.random()])
        k = 8.31e-3
        E_array[i] = -1/2*k*T_0*np.log(l)
    return E_array


@njit('float64(float64, float64)')
def _P_formula(m: float, E_i: float) -> float:
    mult = -1 if np.random.random() < 0.5 else 1
    return mult*np.sqrt(2*m*E_i)


@njit('float64[:,:](int32, float64, float64[:,:])')
def P(N: int, m: float, E: np.array):
    P_array = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        mult0 = -1 if np.random.random() < 0.5 else 1
        mult1 = -1 if np.random.random() < 0.5 else 1
        mult2 = -1 if np.random.random() < 0.5 else 1
        multvec = np.array([mult0, mult1, mult2])
        P_array[i] = multvec*np.sqrt(2*m*E[i])

    return P_array


@njit('float64(float64[:], float64[:], float64, float64)')
def V_P(r_i: np.array, r_j: np.array, epsilon: float, R: float) -> float:
    r_ij = np.linalg.norm(r_i - r_j)
    alpha = (R*R) / (r_ij*r_ij)
    beta = alpha*alpha*alpha
    return epsilon*(beta*beta - 2*beta)


@njit('float64(float64[:], float64, float64)')
def V_S(r_i: np.array, L: float, f: float) -> float:
    r_i_norm = np.linalg.norm(r_i)
    if r_i_norm < L:
        return 0.0
    res = f/2*(r_i_norm - L)*(r_i_norm - L)
    return res


@njit('float64(float64[:,:], float64, float64, float64, float64)', parallel=True)
def V(r: np.array, epsilon: float, R: float, L: float, f: float) -> float:
    N = len(r)

    # print(r)
    V_P_sum = 0
    for i in range(N):
        for j in range(i):
            if i == j:
                continue
            V_P_temp = V_P(r[i], r[j], epsilon, R)
            V_P_sum += V_P_temp

    V_S_sum = 0
    for i in range(N):
        V_S_sum += V_S(r[i], L, f)

    return V_P_sum + V_S_sum


@njit('float64[:](float64[:], float64[:], float64, float64)')
def _F_P_formula(r_i, r_j, epsilon, R):
    r_ij = np.linalg.norm(r_i - r_j)
    alpha = (R*R) / (r_ij*r_ij)
    beta = alpha * alpha * alpha
    return 12*epsilon * (beta * beta - beta)*(r_i - r_j)/(r_ij*r_ij)


@njit('float64[:,:, :](int32, float64[:,:], float64, float64)')
def _F_P_array(N, r, epsilon, R):
    F_P_arr = np.empty((N, N, 3), dtype=np.float64)
    for i in range(N):
        for j in range(i):
            if i == j:
                F_P_arr[i][j] = np.array([0., 0., 0.])
                continue
            F_P_arr[i][j] = _F_P_formula(r[i], r[j], epsilon, R)
            F_P_arr[j][i] = -F_P_arr[i][j]

    return F_P_arr


@njit('float64[:](int32,  int32, float64[:,:,:])')
def _F_P_sum(N, i, F_P_arr):
    F_P_sum = np.zeros(3, dtype=np.float64)
    for j in range(N):
        if i == j:
            continue
        F_P_sum += F_P_arr[i][j]
    return F_P_sum


@njit('float64[:,:](float64[:,:], float64, float64)')
def F_P(r, epsilon: float, R: float) -> np.array:
    N = len(r)
    F_P_array = np.zeros((N, 3), dtype=np.float64)
    F_P_atom_arr = _F_P_array(N, r, epsilon, R)
    for i in range(N):
        F_P_array[i] = _F_P_sum(N, i, F_P_atom_arr)
    return F_P_array


@njit('float64[:](float64[:], float64, float64)')
def _F_S_formula(r_i: np.array, L: float, f: float):
    r_i_norm = np.linalg.norm(r_i)
    if r_i_norm < L:
        return np.array([0., 0., 0.])
    res = f*(L-r_i_norm)*r_i/r_i_norm
    return res


@njit('float64[:,:](float64[:,:], float64, float64)')
def F_S(r: np.array, L: float, f: float) -> np.array:
    F_S_array = np.empty((len(r), 3))
    for i in range(len(r)):
        F_S_array[i] = _F_S_formula(r[i], L, f)
    return F_S_array


@njit('float64(float64[:,:], float64)')
def p(F_S: np.array, L: float) -> np.array:
    F_S_sum = np.empty((3), dtype=np.float64)
    F_S_x = [f[0] for f in F_S]
    F_S_y = [f[1] for f in F_S]
    F_S_z = [f[2] for f in F_S]
    F_S_sum[0] = np.sum(np.array(F_S_x))
    F_S_sum[1] = np.sum(np.array(F_S_y))
    F_S_sum[2] = np.sum(np.array(F_S_z))
    return 1/(4*np.pi*L*L)*np.linalg.norm(F_S_sum)


@njit('float64(float64[:,:], float64, float64)')
def H(p: np.array, m: float, V: float):
    return np.sum(np.array([norm(p_i)*norm(p_i)/(2*m) for p_i in p])) + V


@njit('float64[:,:](float64[:,:], float64[:,:], float64)')
def P_advance(p: np.array, F: np.array, tau: float):
    p_new = p + F/2 * tau
    return p_new


@njit('float64[:,:](float64[:,:], float64, float64[:,:], float64)')
def r_advance(r, m, p, tau):
    r_new = r + p/m * tau
    return r_new


# (19)
@njit('float64[:,:], float64')
def T(p, m):
    k = 8.31e-3
    N = len(p)
    p_sum = np.sum(np.array([norm(p_i)*norm(p_i)/(2*m) for p_i in p]))
    return 2/(3*N*k) * p_sum
