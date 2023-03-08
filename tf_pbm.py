import tensorflow as tf
import numpy as np
import asset


class InternalExchangeLayer(tf.keras.layers.Layer):
    def __init__(self, R_inv_internal_on, R_inv_internal_off=None, trainable=True, name=None):
        super(InternalExchangeLayer, self).__init__()
        num_rooms = R_inv_internal_on.shape[0]
        self.num_rooms = num_rooms
        # TODO: Ensure that  the matrixes are symmetric
        self.R_inv_combination = InputCombinationLayer(R_inv_internal_on, R_inv_internal_off, trainable=trainable, name="R_inv_internal")
    
    def call(self, T_rooms, R_is_on):
        T_rooms_tiled = tf.tile(T_rooms, [1,1, self.num_rooms])
        T_rooms_tiled_tran = tf.transpose(T_rooms_tiled, perm=[0, 2, 1])
        skew_diff = T_rooms_tiled_tran - T_rooms_tiled
        R_inv_internal = self.R_inv_combination(R_is_on)
        individual_exchange = tf.keras.layers.Multiply()([R_inv_internal, skew_diff])
        reduce_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=2, keepdims=True), name="reduce_sum")
        summed_exchange = reduce_sum(individual_exchange)
        return summed_exchange

class TemperatureUpdateLayer(tf.keras.layers.Layer):
    def __init__(self, C_inv, delta_t, trainable=True, name=None):
        super(TemperatureUpdateLayer, self).__init__()
        self.delta_t = tf.constant(delta_t, dtype=tf.float32)
        self.C_inv = tf.Variable(C_inv.reshape((1, *C_inv.shape)), dtype=tf.float32, trainable=trainable, constraint=tf.keras.constraints.NonNeg())
    
    def call(self, T, rhs):

        change = tf.keras.layers.Multiply()([self.C_inv, rhs])
        T_new = tf.keras.layers.Add()([T , self.delta_t*change])
        return T_new

class DirectExchangeLayer(tf.keras.layers.Layer):
    def __init__(self, R_inv_on, R_inv_off, trainable=True, name=None):
        super(DirectExchangeLayer, self).__init__()
        self.R_combination = InputCombinationLayer(R_inv_on, R_inv_off, trainable=trainable, name="R_inv_outside")
    
    def call(self, T_rooms, T_compare, is_on):
        diff = T_compare - T_rooms
        R_inv = self.R_combination(is_on)
        exchange = tf.keras.layers.Multiply()([R_inv, diff])
        return exchange

class WallOutsideExchangeLayer(tf.keras.layers.Layer):
    def __init__(self, R_inv_out_wall_outside, R_inv_out_wall, u_outside_on, u_outside_off, trainable=True, u_gains_trainable=True, R_walls_trainable=True, name=None):
        super(WallOutsideExchangeLayer, self).__init__()
        self.num_rooms = R_inv_out_wall_outside.shape[0]
        self.R_inv_out_wall_outside = tf.Variable(R_inv_out_wall_outside.reshape((1, *R_inv_out_wall_outside.shape)), dtype=tf.float32, trainable=trainable and R_walls_trainable, constraint=tf.keras.constraints.NonNeg())
        self.R_inv_out_wall = tf.Variable(R_inv_out_wall.reshape((1, *R_inv_out_wall.shape)), dtype=tf.float32, trainable=trainable and R_walls_trainable, constraint=tf.keras.constraints.NonNeg())
        self.u_outside = InputCombinationLayer(u_outside_on, u_outside_off, trainable=trainable and u_gains_trainable, name="u_outside")
        self.divide = tf.math.divide_no_nan

    def call(self, T_wall, T_out, u_is_on):

        u_outside = self.u_outside(u_is_on)
        u_outside_sum = tf.reduce_sum(u_outside, axis=2, keepdims=True)

        diff = T_out - T_wall
        R_prod = self.R_inv_out_wall*self.R_inv_out_wall_outside
        R_sum = self.R_inv_out_wall + self.R_inv_out_wall_outside
        scaled_diff = R_prod * diff

        out_wall_to_wall_exchange = self.divide(self.R_inv_out_wall* u_outside_sum + scaled_diff, R_sum)

        return out_wall_to_wall_exchange


class RoomWallExchange(tf.keras.layers.Layer):
    def __init__(self, R_inv_in_wall, R_inv_room_wall, u_inside_on, u_inside_off, trainable=True, u_gains_trainable=True, R_walls_trainable=True, name=None):
        super(RoomWallExchange, self).__init__()
        self.num_rooms = R_inv_in_wall.shape[0]
        self.R_inv_in_wall = tf.Variable(R_inv_in_wall.reshape((1, *R_inv_in_wall.shape)), dtype=tf.float32, trainable=trainable and R_walls_trainable, constraint=tf.keras.constraints.NonNeg())
        self.R_inv_room_wall = tf.Variable(R_inv_room_wall.reshape((1, *R_inv_room_wall.shape)), dtype=tf.float32, trainable=trainable and R_walls_trainable, constraint=tf.keras.constraints.NonNeg())
        self.u_inside = InputCombinationLayer(u_inside_on, u_inside_off, trainable=trainable and u_gains_trainable, name="u_inside")
        self.divide = tf.math.divide_no_nan

    

    def call(self, T_rooms, T_wall, u_is_on):

        u_inside = self.u_inside(u_is_on)
        u_inside_sum = tf.reduce_sum(u_inside, axis=2, keepdims=True)

        in_wall_R_prod_T_diff = self.R_inv_in_wall * self.R_inv_room_wall * (T_rooms - T_wall)
        in_wall_R_sum = self.R_inv_in_wall + self.R_inv_room_wall

        in_wall_to_wall_exchange = self.divide(self.R_inv_in_wall * u_inside_sum + in_wall_R_prod_T_diff, in_wall_R_sum)

        in_wall_to_room_exchange = self.divide(self.R_inv_room_wall * u_inside_sum - in_wall_R_prod_T_diff, in_wall_R_sum)

        return in_wall_to_wall_exchange, in_wall_to_room_exchange

class LinearCombinationLayer(tf.keras.layers.Layer):
    def __init__(self, on_gain, off_gain, trainable=True):
        super(LinearCombinationLayer, self).__init__()
        self.on_gain = tf.Variable(on_gain.reshape(1, *on_gain.shape), dtype=tf.float32, trainable=trainable, constraint=tf.keras.constraints.NonNeg())
        self.off_gain = tf.Variable(off_gain.reshape(1, *off_gain.shape), dtype=tf.float32, trainable=trainable, constraint=tf.keras.constraints.NonNeg())
    
    def call(self, x):
        return tf.keras.layers.Add(name='LinearCombinationLayer')([tf.keras.layers.Multiply()([self.on_gain, x]), tf.keras.layers.Multiply()([self.off_gain, 1-x])])

class InverseLinearCombinationLayer(tf.keras.layers.Layer):
    def __init__(self, on_gain, off_gain, trainable=True):
        super(InverseLinearCombinationLayer, self).__init__()
        self.on_gain = tf.Variable(on_gain.reshape(1, *on_gain.shape), dtype=tf.float32, trainable=trainable, constraint=tf.keras.constraints.NonNeg())
        self.off_gain = tf.Variable(off_gain.reshape(1, *off_gain.shape), dtype=tf.float32, trainable=trainable, constraint=tf.keras.constraints.NonNeg())
        self.divide = tf.math.divide_no_nan

    def call(self, x):
        linear = tf.keras.layers.Add()([tf.keras.layers.Multiply()([self.on_gain, 1-x]), tf.keras.layers.Multiply()([self.off_gain, x])])
        prod = tf.keras.layers.Multiply()([self.on_gain, self.off_gain])
        return self.divide(prod, linear)
    
class VariableReturnLayer(tf.keras.layers.Layer):
    def __init__(self, var, trainable=True):
        super(VariableReturnLayer, self).__init__()
        self.var = tf.Variable(var.reshape(1, *var.shape), dtype=tf.float32, trainable=trainable, constraint=tf.keras.constraints.NonNeg())
    
    def call(self, x):
        return self.var

class InputCombinationLayer(tf.keras.layers.Layer):
    def __init__(self, on_gain, off_gain=None, trainable=True, inverse=False, name=None):
        super(InputCombinationLayer, self).__init__()
        if off_gain is None:
            self.combination = VariableReturnLayer(on_gain, trainable=trainable)
        elif inverse:
            self.combination = InverseLinearCombinationLayer(on_gain, off_gain, trainable=trainable)
        else:
            self.combination = LinearCombinationLayer(on_gain, off_gain, trainable=trainable)
        
    
    def call(self, input_is_on):
        return self.combination(input_is_on)


class ThermoPBMLayer(tf.keras.layers.Layer):
    def __init__(self, 
        num_rooms: int,
        delta_t: float, 
        num_u = 1,
        num_direct_connections = 1,

        variable_R_inv_internal=False,
        R_inv_internal=None, 
        
        R_inv_walls=None, 
        R_inv_out_direct_connections=None,
        C_inv_rooms=None, 
        C_inv_walls=None, 
        u_gains=None,

        trainable=True,

        R_walls_trainable=True,
        C_walls_trainable=True,
        u_gains_trainable=True,
        R_internal_trainable=True,
        C_rooms_trainable=True,
        R_direct_connections_trainable=True,

        name=None,
        ):
        super(ThermoPBMLayer, self).__init__()
        self.num_rooms = num_rooms
        self.num_u = num_u
        self.num_direct_connections = num_direct_connections

        if R_inv_internal is None:
            num_R_internal = 2 if variable_R_inv_internal else 1
            R_inv_internal_list = [None, None]
            for i in range(num_R_internal):
                random_matrix =1/(np.random.rand(num_rooms, num_rooms)*99 + 1)
                random_symmetric_matrix = (random_matrix + random_matrix.T)/2
                random_symmetric_matrix_with_zero_diagonal = random_symmetric_matrix - np.diag(np.diag(random_symmetric_matrix))
                R_inv_internal_list[i] = random_symmetric_matrix_with_zero_diagonal
        
        if R_inv_walls is None:
            R_inv_walls = 1/(np.random.rand(num_rooms, 4)*99 + 1)
        
        if R_inv_out_direct_connections is None:
            R_inv_out_direct_connections = 1/(np.random.rand(num_direct_connections, 2, num_rooms, 1)*99 + 1)

        if C_inv_rooms is None:
            C_inv_rooms = 1/(np.random.rand(num_rooms, 1)*99 + 1)
        
        if C_inv_walls is None:
            C_inv_walls = 1/(np.random.rand(num_rooms, 1)*99 + 1)
        
        if u_gains is None:
            u_gains = np.random.rand(num_rooms, num_u, 6)*9 + 1

        R_inv_out_wall_outside = R_inv_walls[:,0].reshape(-1,1)
        R_inv_out_wall = R_inv_walls[:,1].reshape(-1,1)
        R_inv_in_wall = R_inv_walls[:,2].reshape(-1,1)
        R_inv_room_wall = R_inv_walls[:,3].reshape(-1,1)
        
        u_direct_on = u_gains[:,:,0]
        u_direct_off = u_gains[:,:,1]
        u_inside_on = u_gains[:,:,2]
        u_inside_off = u_gains[:,:,3]
        u_outside_on = u_gains[:,:,4]
        u_outside_off = u_gains[:,:,5]


        self.u_direct = InputCombinationLayer(u_direct_on, u_direct_off, trainable=trainable and u_gains_trainable, name='u_direct')

        self.internal_exchange = InternalExchangeLayer(R_inv_internal_list[0], R_inv_internal_list[1], trainable=trainable and R_internal_trainable)
        
        self.room_wall_exchange = RoomWallExchange(R_inv_in_wall, R_inv_room_wall, u_inside_on, u_inside_off, trainable=trainable, R_walls_trainable=R_walls_trainable, u_gains_trainable=u_gains_trainable)
        
        self.wall_outside_exchange = WallOutsideExchangeLayer(R_inv_out_wall_outside, R_inv_out_wall, u_outside_on, u_outside_off, trainable=trainable, R_walls_trainable=R_walls_trainable, u_gains_trainable=u_gains_trainable)
        
        self.room_direct_connections_exchange = []
        
        for i in range(num_direct_connections):
            direct_layer = DirectExchangeLayer(R_inv_out_direct_connections[i][0], R_inv_out_direct_connections[i][1], trainable=trainable and R_direct_connections_trainable)
            self.room_direct_connections_exchange.append(direct_layer)
        
        
        self.temperature_update_room = TemperatureUpdateLayer(C_inv_rooms, delta_t, trainable=trainable and C_rooms_trainable)
        self.temperature_update_wall = TemperatureUpdateLayer(C_inv_walls, delta_t, trainable=trainable and C_walls_trainable)

        self.delta_t = tf.constant(delta_t, dtype=tf.float32)

        self.sum_direct_connections = tf.keras.layers.Add() if self.num_direct_connections > 1 else lambda x: x[0]


    
    def call(self, T_rooms, T_wall, T_out, T_direct_connections, u_is_on, internal_exchange_is_on, direct_connections_is_on, corrective_source_term):

        u_direct = self.u_direct(u_is_on)
        u_direct_sum = tf.reduce_sum(u_direct, axis=2, keepdims=True)

        in_wall_to_wall_exchange, in_wall_to_room_exchange = self.room_wall_exchange(T_rooms, T_wall, u_is_on)
        out_wall_to_wall_exchange = self.wall_outside_exchange(T_wall, T_out, u_is_on)
        
       
        internal_exchange = self.internal_exchange(T_rooms, internal_exchange_is_on)

        direct_connections_exchange = []

        for i in range(self.num_direct_connections):
            direct_connections_exchange.append(self.room_direct_connections_exchange[i](T_rooms, T_direct_connections[:,i], direct_connections_is_on[:,i]))


        sum_direct_connections_exchange = self.sum_direct_connections(direct_connections_exchange)

        rhs_rooms = tf.keras.layers.Add()([in_wall_to_room_exchange, sum_direct_connections_exchange, internal_exchange, u_direct_sum, corrective_source_term[:,0]])
        rhs_wall = tf.keras.layers.Add()([in_wall_to_wall_exchange, out_wall_to_wall_exchange, corrective_source_term[:,1]])

        T_rooms_new = self.temperature_update_room(T_rooms, rhs_rooms)
        T_wall_new = self.temperature_update_wall(T_wall, rhs_wall)

        return T_rooms_new, T_wall_new
    

def create_tf_single_step_layer(num_rooms: int, delta_t: float, num_u=1, num_direct_connections=1, variable_R_internal=True,  R_inv_internal=None, R_inv_walls=None, C_inv_rooms=None, C_inv_walls=None, u_gains=None):
    print("Creating tf single step layer")
    


    thermo_layer = ThermoPBMLayer(num_rooms, delta_t,num_u=num_u, num_direct_connections=num_direct_connections, variable_R_inv_internal=variable_R_internal, R_inv_internal=R_inv_internal, R_inv_walls=R_inv_walls, C_inv_rooms=C_inv_rooms, C_inv_walls=C_inv_walls, u_gains=u_gains)
    
    
    T_rooms = tf.keras.Input(shape=(num_rooms,1), name="T_rooms")
    T_wall = tf.keras.Input(shape=(num_rooms,1), name="T_wall")

    T_out = tf.keras.Input(shape=(1), name="T_out")

    T_direct_connections = tf.keras.Input(shape=(num_direct_connections,1), name="T_direct_connections")

    u_is_on = tf.keras.Input(shape=(num_rooms, num_u), name="u_is_on")
    internal_exchange_is_on = tf.keras.Input(shape=(num_rooms, num_rooms), name="internal_exchange_is_on")
    direct_connection_is_on = tf.keras.Input(shape=(num_direct_connections, num_rooms, 1), name="ventilation_is_on")

    corrective_source_term = tf.keras.Input(shape=(2,num_rooms,1), name="corrective_source_term")

    inputs = [T_rooms, T_wall, T_out, T_direct_connections, u_is_on, internal_exchange_is_on, direct_connection_is_on, corrective_source_term]


    

    T_new_rooms, T_new_wall = thermo_layer.call(*inputs)

    return tf.keras.Model(inputs=inputs, outputs=[T_new_rooms, T_new_wall])

if __name__ == '__main__':
    #Asset values
    sim_asset = asset.get_asset()
    R_inv_internal = sim_asset.get_R_partWall_open_inv()
    
    R_inv_wall = sim_asset.get_R_inv()

    C_inv_rooms = sim_asset.get_C_open_inv()[0]
    C_inv_walls = sim_asset.get_C_open_inv()[1]

    num_rooms = C_inv_rooms.size

    R_inv_wall = np.hstack((R_inv_wall, np.zeros((num_rooms, 1))))


    #Initial values
    T_rooms, T_wall, T_out = asset.get_initial_values()

    u_gains = np.array([[0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0]])

    delta_t = 1e-2

    tf_single_step_layer = create_tf_single_step_layer(num_rooms, delta_t, num_u=2, num_direct_connections=2, variable_R_internal=True )
    tf_single_step_layer.summary()

    T_new_rooms, T_new_wall = T_rooms, T_wall

    T_direct_connections = np.array([[5, 0]])

    u_is_on = np.array([[[0, 1],
                         [1, 0],
                         [0, 0]]])

    direct_connection_is_on = np.array([[[[0], [0], [1]],
                                        [[1], [1],[0]]]])

    R_internal_on = np.array([[[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]]])
    
    corrective_source_term = np.array([ [ [[0],[0],[0]], [[0],[0],[0]] ] ])

    T = 1
    for i in range(int(T/delta_t)):
        T_new_rooms, T_new_wall = tf_single_step_layer([T_new_rooms, T_new_wall, T_out, T_direct_connections, u_is_on, R_internal_on, direct_connection_is_on, corrective_source_term])

    print(T_new_rooms)
    print(T_new_wall)