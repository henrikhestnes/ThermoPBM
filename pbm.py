import numpy as np
import tensorflow as tf

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


def get_rhs(T_rooms: np.ndarray,T_wall: np.ndarray,T_out: float, R_inv_internal: np.ndarray, R_inv_out_wall:np.ndarray, R_inv_outside:np.ndarray,u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
        T_rooms: 1D array of air temperatures of the rooms
        T_wall: 1D array of temperatures of the walls
        T_out: temperature of the outside

        R_inv_internal: 2D array of inverse resistances between rooms (symetric matrix with zero on diagonal)
        R_inv_wall: 2D array of inverse resistances charaterizing the wall, with the following structure:
            R_inv_out_wall: inverse resistance between the wall and the outside
            R_inv_in_wall: inverse resistance between the wall and the inside
            R_inv_room_wall: inverse resistance between the wall and the rooms
        R_inv_outside: inverse resistance between the outside and the rooms in direct connection (e.g ventilation)

        u: 2D array of source terms for the rooms, with the following structure:
            u_out_wall: source term for heating the wall from the outside
            u_in_wall: source term for heating the wall from the inside
            u_direct: source term for heating directly in the rooms
    """

    internal_room_exchange = exchange_between_nodes(T_rooms, R_inv_internal)
    
    R_inv_out_wall_outside = R_inv_out_wall[:,0].reshape(-1,1)
    R_inv_out_wall = R_inv_wall[:,1].reshape(-1,1)
    R_inv_in_wall = R_inv_wall[:,2].reshape(-1,1)
    R_inv_room_wall = R_inv_wall[:,3].reshape(-1,1)

    u_out_wall = u[:,0].reshape(-1,1)
    u_in_wall = u[:,1].reshape(-1,1)
    u_direct = u[:,2].reshape(-1,1)

    directly_outside_exchange = R_inv_outside * (T_out - T_rooms)

    R_inv_out_wall_sum = R_inv_out_wall + R_inv_out_wall_outside
    R_inv_out_wall_sum[R_inv_out_wall_sum == 0] = 1

    out_wall_to_wall_exchange = (R_inv_out_wall * u_out_wall + R_inv_out_wall*R_inv_out_wall_outside*(T_out - T_wall))/R_inv_out_wall_sum

    in_wall_R_prod_T_diff = np.multiply(R_inv_in_wall*R_inv_room_wall, (T_rooms - T_wall))
    in_wall_R_sum = R_inv_in_wall + R_inv_room_wall
    in_wall_R_sum[in_wall_R_sum == 0] = 1

    in_wall_to_wall_exchange = (R_inv_in_wall * u_in_wall + in_wall_R_prod_T_diff)/in_wall_R_sum

    in_wall_to_room_exchange = (R_inv_room_wall * u_in_wall - in_wall_R_prod_T_diff)/in_wall_R_sum

    rhs_rooms = internal_room_exchange + directly_outside_exchange + in_wall_to_room_exchange + u_direct 
    rhs_wall = out_wall_to_wall_exchange + in_wall_to_wall_exchange
    
    return rhs_rooms, rhs_wall

def step(T_rooms: np.ndarray, T_wall: np.ndarray, T_out:float,R_inv_internal: np.ndarray, R_inv_wall:np.ndarray, R_inv_outside:np.ndarray,u: np.ndarray,  C_inv_rooms: np.ndarray, C_inv_wall: np.ndarray, delta_t: float) -> tuple[np.ndarray, np.ndarray]:
    rhs_rooms, rhs_wall = get_rhs(T_rooms, T_wall, T_out,R_inv_internal, R_inv_wall, R_inv_outside,u)
    T_new_rooms = T_rooms + delta_t*C_inv_rooms*rhs_rooms
    T_new_wall = T_wall + delta_t*C_inv_wall*rhs_wall
    return T_new_rooms, T_new_wall


if __name__ == "__main__":
    T_rooms = np.array([[3], [3], [3]])
    T_wall = np.array([[1], [1], [1]])
    T_out = 0


    R_inv_internal = np.array([[0, 10, 0], 
                [10, 0, 1], 
                [0, 1, 0]])
    
    R_inv_wall = np.array([
        [0, 0, 10, 10], 
        [0, 0.1, 10, 10], 
        [10, 10., 1, 1]])
    R_inv_outside = np.array([[0], [0], [0]])

    C_inv_rooms = np.array([[1], [1], [1]])
    C_inv_walls = np.array([[1], [1], [1]])
 
    u = np.array([
        [0,0,0], 
        [0,0,0], 
        [10,0,0]
        ])

    delta_t = 1e-2
    N = 100
    for i in range(N):
        T_rooms, T_wall = step(T_rooms, T_wall, T_out,R_inv_internal, R_inv_wall, R_inv_outside,u,  C_inv_rooms, C_inv_walls, delta_t)
        print(T_rooms)
        print(T_wall)



