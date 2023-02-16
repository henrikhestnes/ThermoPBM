import tensorflow as tf
import numpy as np
import asset


class InternalExchangeLayer(tf.keras.layers.Layer):
    def __init__(self, num_rooms, R_inv_internal):
        super(InternalExchangeLayer, self).__init__()
        self.num_rooms = num_rooms
        self.R_inv_internal = tf.Variable(R_inv_internal.reshape((1, *R_inv_internal.shape)), dtype=tf.float32, trainable=True)

    def call(self, inputs):
        T_rooms = inputs
        T_rooms_tiled = tf.tile(T_rooms, [1,1, self.num_rooms])
        T_rooms_tiled_tran = tf.transpose(T_rooms_tiled, perm=[0, 2, 1])
        skew_diff = T_rooms_tiled_tran - T_rooms_tiled
        individual_exchange = tf.keras.layers.Multiply()([self.R_inv_internal, skew_diff])
        reduce_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=2, keepdims=True), name="reduce_sum")
        summed_exchange = reduce_sum(individual_exchange)
        return summed_exchange

class TemperatureUpdateLayer(tf.keras.layers.Layer):
    def __init__(self, C_inv, delta_t):
        super(TemperatureUpdateLayer, self).__init__()
        self.delta_t = tf.constant(delta_t, dtype=tf.float32)
        self.C_inv = tf.Variable(C_inv.reshape((1, *C_inv.shape)), dtype=tf.float32, trainable=True)
    
    def call(self, inputs):
        T = inputs[0]
        rhs = inputs[1]
        change = tf.math.multiply(self.C_inv, rhs)

        T_new = tf.keras.layers.Add(name='T_out')([T , self.delta_t*change])
        return T_new

class DirectOutsideExchangeLayer(tf.keras.layers.Layer):
    def __init__(self, R_inv_outside):
        super(DirectOutsideExchangeLayer, self).__init__()
        self.R_inv_outside = tf.Variable(R_inv_outside.reshape((1, *R_inv_outside.shape)), dtype=tf.float32, trainable=True)

    def call(self, inputs):
        T_rooms = inputs[0]
        T_out = inputs[1]
        diff = T_out - T_rooms
        exchange = tf.keras.layers.Multiply()([self.R_inv_outside, diff])
        return exchange

class WallOutsideExchangeLayer(tf.keras.layers.Layer):
    def __init__(self, R_inv_out_wall_outside, R_inv_out_wall):
        super(WallOutsideExchangeLayer, self).__init__()
        
        self.R_inv_out_wall_outside = tf.Variable(R_inv_out_wall_outside.reshape((1, *R_inv_out_wall_outside.shape)), dtype=tf.float32, trainable=True)
        self.R_inv_out_wall = tf.Variable(R_inv_out_wall.reshape((1, *R_inv_out_wall.shape)), dtype=tf.float32, trainable=True)

    def call(self, inputs):
        T_wall = inputs[0]
        T_out = inputs[1]
        u_outside = inputs[2]

        out_wall_to_wall_exchange = (self.R_inv_out_wall * u_outside + self.R_inv_out_wall*self.R_inv_out_wall_outside*(T_out - T_wall))/(self.R_inv_out_wall + self.R_inv_out_wall_outside)

        return out_wall_to_wall_exchange


class RoomWallExchange(tf.keras.layers.Layer):
    def __init__(self, R_inv_in_wall, R_inv_room_wall):
        super(RoomWallExchange, self).__init__()
        self.R_inv_in_wall = tf.Variable(R_inv_in_wall.reshape((1, *R_inv_in_wall.shape)), dtype=tf.float32, trainable=True)
        self.R_inv_room_wall = tf.Variable(R_inv_room_wall.reshape((1, *R_inv_room_wall.shape)), dtype=tf.float32, trainable=True)

    def call(self, inputs):
        T_rooms = inputs[0]
        T_wall = inputs[1]
        u_inside = inputs[2]

        in_wall_R_prod_T_diff = self.R_inv_in_wall * self.R_inv_room_wall * (T_rooms - T_wall)
        in_wall_R_sum = self.R_inv_in_wall + self.R_inv_room_wall

        in_wall_to_wall_exchange = (self.R_inv_in_wall * u_inside + in_wall_R_prod_T_diff)/in_wall_R_sum

        in_wall_to_room_exchange = (self.R_inv_room_wall * u_inside - in_wall_R_prod_T_diff)/in_wall_R_sum

        return in_wall_to_wall_exchange, in_wall_to_room_exchange


def create_tf_single_step_layer(num_rooms: int, R_inv_internal, R_inv_walls, C_inv_rooms, C_inv_walls, delta_t: float):
    print("Creating tf single step layer")
    T_rooms = tf.keras.Input(shape=(num_rooms,1), name="T_rooms")
    T_wall = tf.keras.Input(shape=(num_rooms,1), name="T_wall")

    T_out = tf.keras.Input(shape=(1), name="T_out")
    u_outside = tf.keras.Input(shape=(num_rooms,1), name="u_outside")
    u_inside = tf.keras.Input(shape=(num_rooms,1), name="u_inside")
    u_direct = tf.keras.Input(shape=(num_rooms,1), name="u_direct")

    R_inv_out_wall_outside = R_inv_walls[:,0].reshape(-1,1)
    R_inv_out_wall = R_inv_walls[:,1].reshape(-1,1)
    R_inv_in_wall = R_inv_walls[:,2].reshape(-1,1)
    R_inv_room_wall = R_inv_walls[:,3].reshape(-1,1)
    R_inv_outside = R_inv_walls[:,4].reshape(-1,1)

    internal_exchange_layer = InternalExchangeLayer(num_rooms, R_inv_internal)
    direct_outside_exchange_layer = DirectOutsideExchangeLayer(R_inv_outside)
    wall_outside_exchange_layer = WallOutsideExchangeLayer(R_inv_out_wall_outside, R_inv_out_wall)
    room_wall_exchange_layer = RoomWallExchange(R_inv_in_wall, R_inv_room_wall)

    
    internal = internal_exchange_layer(T_rooms)

    directly_outside = direct_outside_exchange_layer([T_rooms, T_out])

    out_wall_to_wall = wall_outside_exchange_layer([T_wall, T_out, u_outside])
    
    in_wall_to_wall, in_wall_to_room = room_wall_exchange_layer([T_rooms, T_wall, u_inside])

    rhs_rooms = tf.keras.layers.Add(name='RHS_rooms')([internal, in_wall_to_room, directly_outside, u_direct])
    rhs_walls = tf.keras.layers.Add(name='RHS_walls')([out_wall_to_wall, in_wall_to_wall])

    T_new_rooms = TemperatureUpdateLayer(C_inv_rooms, delta_t)([T_rooms, rhs_rooms])
    T_new_wall = TemperatureUpdateLayer(C_inv_walls, delta_t)([T_wall, rhs_walls])

    return tf.keras.Model(inputs=[T_rooms, T_wall, T_out, u_outside, u_inside, u_direct], outputs=[T_new_rooms, T_new_wall])

if __name__ == '__main__':
    #Asset values
    sim_asset = asset.get_asset()
    R_inv_internal = sim_asset.R_partWall_inv
    
    R_inv_wall = sim_asset.get_R_inv()

    C_inv_rooms = sim_asset.C_room_inv
    C_inv_walls = sim_asset.C_wall_inv


    #Initial values
    T_rooms = np.array([[[3], [3], [3]]])
    T_wall = np.array([[[1], [1], [1]]])
    T_out = np.array([[1]])

    u_outside = np.array([[[0], [0], [0]]])
    u_inside = np.array([[[0], [0], [0]]])
    u_direct = np.array([[[0], [0], [0]]])

    delta_t = 1e-2

    tf_single_step_layer = create_tf_single_step_layer(3, R_inv_internal, R_inv_wall, C_inv_rooms, C_inv_walls, delta_t)
    tf_single_step_layer.summary(line_length=150)

    T_new_rooms, T_new_wall = T_rooms, T_wall

    T = 1
    for i in range(int(T/delta_t)):
        T_new_rooms, T_new_wall = tf_single_step_layer([T_new_rooms, T_new_wall, T_out, u_outside, u_inside, u_direct])

    print(T_new_rooms)
    print(T_new_wall)