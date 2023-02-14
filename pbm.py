import numpy as np

def skew_difference_matrix(T: np.ndarray) -> np.ndarray:
    n = T.size
    T_stack = np.tile(T, (1,n))
    diff = T_stack - T_stack.T
    return -diff

def exchange_between_nodes(T: np.ndarray, A: np.ndarray) -> np.ndarray:
    diff = skew_difference_matrix(T)
    individual_exchange = np.multiply(A, diff)
    net = np.sum(individual_exchange, axis=1, keepdims=True)
    return net

def get_rhs(T: np.ndarray, A: np.ndarray, C_inv: np.ndarray) -> np.ndarray:
    exchange = exchange_between_nodes(T, A)
    rhs = np.multiply(C_inv, exchange)
    return rhs

def step(T: np.ndarray, A: np.ndarray, C_inv: np.ndarray, delta_t: float) -> np.ndarray:
    rhs = get_rhs(T, A, C_inv)
    T_new = T + delta_t*rhs
    return T_new

if __name__ == "__main__":
    T = np.array([[0], [3], [3]])


    A = np.array([[0, 1, 1], 
                [1, 0, 1], 
                [1, 1, 0]])

    C_inv = np.array([[1], [1], [1]])
    delta_t = 1e-3
    N = 100
    for i in range(N):
        T = step(T, A, C_inv, delta_t)
        print(T)



