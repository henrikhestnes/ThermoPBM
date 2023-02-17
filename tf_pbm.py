import tensorflow as tf
import numpy as np
import asset


class InternalExchangeLayer(tf.keras.layers.Layer):
    def __init__(self, R_inv_internal, trainable=True, name=None):
        super(InternalExchangeLayer, self).__init__()
        num_rooms = R_inv_internal.shape[0]
        self.num_rooms = num_rooms
        self.R_inv_internal = tf.Variable(R_inv_internal.reshape((1, *R_inv_internal.shape)), dtype=tf.float32, trainable=trainable, constraint=tf.keras.constraints.NonNeg())

    def call(self, T_rooms):
        T_rooms_tiled = tf.tile(T_rooms, [1,1, self.num_rooms])
        T_rooms_tiled_tran = tf.transpose(T_rooms_tiled, perm=[0, 2, 1])
        skew_diff = T_rooms_tiled_tran - T_rooms_tiled
        individual_exchange = tf.keras.layers.Multiply()([self.R_inv_internal, skew_diff])
        reduce_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=2, keepdims=True), name="reduce_sum")
        summed_exchange = reduce_sum(individual_exchange)
        return summed_exchange

class TemperatureUpdateLayer(tf.keras.layers.Layer):
    def __init__(self, C_inv, delta_t, trainable=True, name=None):
        super(TemperatureUpdateLayer, self).__init__()
        self.delta_t = tf.constant(delta_t, dtype=tf.float32)
        self.C_inv = tf.Variable(C_inv.reshape((1, *C_inv.shape)), dtype=tf.float32, trainable=trainable, constraint=tf.keras.constraints.NonNeg())
    
    def call(self, T, rhs):

        change = tf.math.multiply(self.C_inv, rhs)
        T_new = tf.keras.layers.Add()([T , self.delta_t*change])
        return T_new

class DirectOutsideExchangeLayer(tf.keras.layers.Layer):
    def __init__(self, R_inv_outside_ventilation, R_inv_outside_no_ventilation, trainable=True, name=None):
        super(DirectOutsideExchangeLayer, self).__init__()
        self.R_combination = InputCombinationLayer(R_inv_outside_ventilation, R_inv_outside_no_ventilation, trainable=trainable, name="R_inv_outside")
    
    def call(self, T_rooms, T_out, ventilation_is_on):
        diff = T_out - T_rooms
        R_inv_outside = self.R_combination(ventilation_is_on)
        exchange = tf.keras.layers.Multiply()([R_inv_outside, diff])
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

        diff = T_out - T_wall
        R_prod = self.R_inv_out_wall*self.R_inv_out_wall_outside
        R_sum = self.R_inv_out_wall + self.R_inv_out_wall_outside
        scaled_diff = R_prod * diff

        out_wall_to_wall_exchange = self.divide(self.R_inv_out_wall* u_outside + scaled_diff, R_sum)

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

        in_wall_R_prod_T_diff = self.R_inv_in_wall * self.R_inv_room_wall * (T_rooms - T_wall)
        in_wall_R_sum = self.R_inv_in_wall + self.R_inv_room_wall

        in_wall_to_wall_exchange = self.divide(self.R_inv_in_wall * u_inside + in_wall_R_prod_T_diff, in_wall_R_sum)

        in_wall_to_room_exchange = self.divide(self.R_inv_room_wall * u_inside - in_wall_R_prod_T_diff, in_wall_R_sum)

        return in_wall_to_wall_exchange, in_wall_to_room_exchange

class InputCombinationLayer(tf.keras.layers.Layer):
    def __init__(self, on_gain, off_gain, trainable=True, name=None):
        super(InputCombinationLayer, self).__init__()
        self.on_gain = tf.Variable(on_gain.reshape(1, *on_gain.shape), dtype=tf.float32, trainable=trainable, constraint=tf.keras.constraints.NonNeg())
        self.off_gain = tf.Variable(off_gain.reshape(1, *off_gain.shape), dtype=tf.float32, trainable=trainable, constraint=tf.keras.constraints.NonNeg())
    
    def call(self, input_is_on):
        return tf.keras.layers.Add(name='input')([tf.keras.layers.Multiply()([self.on_gain, input_is_on]), tf.keras.layers.Multiply()([self.off_gain, 1-input_is_on])])


class ThermoPBMLayer(tf.keras.layers.Layer):
    # Num parameters = num_rooms^2 + 14 * num_rooms
    def __init__(self, 
        num_rooms: int,
        delta_t: float, 
        R_inv_internal=None, 
        R_inv_walls=None, 
        C_inv_rooms=None, 
        C_inv_walls=None, 
        u_gains=None,
        trainable=True,
        R_walls_trainable=True,
        C_walls_trainable=True,
        u_gains_trainable=True,
        R_internal_trainable=True,
        C_rooms_trainable=True,
        R_outside_trainable=True,
        name=None,
        ):
        super(ThermoPBMLayer, self).__init__()
        self.num_rooms = num_rooms

        if R_inv_internal is None:
            random_matrix =1/(np.random.rand(num_rooms, num_rooms)*99 + 1)
            random_symmetric_matrix = (random_matrix + random_matrix.T)/2
            random_symmetric_matrix_with_zero_diagonal = random_symmetric_matrix - np.diag(np.diag(random_symmetric_matrix))
            R_inv_internal = random_symmetric_matrix_with_zero_diagonal
        
        if R_inv_walls is None:
            R_inv_walls = 1/(np.random.rand(num_rooms, 6)*99 + 1)

        if C_inv_rooms is None:
            C_inv_rooms = 1/(np.random.rand(num_rooms, 1)*99 + 1)
        
        if C_inv_walls is None:
            C_inv_walls = 1/(np.random.rand(num_rooms, 1)*99 + 1)
        
        if u_gains is None:
            u_gains = np.random.rand(num_rooms, 6)*9 + 1

        R_inv_out_wall_outside = R_inv_walls[:,0].reshape(-1,1)
        R_inv_out_wall = R_inv_walls[:,1].reshape(-1,1)
        R_inv_in_wall = R_inv_walls[:,2].reshape(-1,1)
        R_inv_room_wall = R_inv_walls[:,3].reshape(-1,1)
        R_inv_outside_ventilation = R_inv_walls[:,4].reshape(-1,1)
        R_inv_outside_no_ventilation = R_inv_walls[:,5].reshape(-1,1)

        u_direct_on = u_gains[:,0].reshape(-1,1)
        u_direct_off = u_gains[:,1].reshape(-1,1)
        u_inside_on = u_gains[:,2].reshape(-1,1)
        u_inside_off = u_gains[:,3].reshape(-1,1)
        u_outside_on = u_gains[:,4].reshape(-1,1)
        u_outside_off = u_gains[:,5].reshape(-1,1)


        self.u_direct = InputCombinationLayer(u_direct_on, u_direct_off, trainable=trainable and u_gains_trainable, name='u_direct')

        self.internal_exchange = InternalExchangeLayer(R_inv_internal, trainable=trainable and R_internal_trainable)
        self.room_wall_exchange = RoomWallExchange(R_inv_in_wall, R_inv_room_wall, u_inside_on, u_inside_off, trainable=trainable, R_walls_trainable=R_walls_trainable, u_gains_trainable=u_gains_trainable)
        self.wall_outside_exchange = WallOutsideExchangeLayer(R_inv_out_wall_outside, R_inv_out_wall, u_outside_on, u_outside_off, trainable=trainable, R_walls_trainable=R_walls_trainable, u_gains_trainable=u_gains_trainable)
        self.room_outside_exchange = DirectOutsideExchangeLayer(R_inv_outside_ventilation, R_inv_outside_no_ventilation, trainable=trainable and R_outside_trainable)

        self.temperature_update_room = TemperatureUpdateLayer(C_inv_rooms, delta_t, trainable=trainable and C_rooms_trainable)
        self.temperature_update_wall = TemperatureUpdateLayer(C_inv_walls, delta_t, trainable=trainable and C_walls_trainable)

        self.delta_t = tf.constant(delta_t, dtype=tf.float32)


    
    def call(self, T_rooms, T_wall, T_out, u_is_on, ventilation_is_on):

        u_direct = self.u_direct(u_is_on)

        in_wall_to_wall_exchange, in_wall_to_room_exchange = self.room_wall_exchange(T_rooms, T_wall, u_is_on)
        out_wall_to_wall_exchange = self.wall_outside_exchange(T_wall, T_out, ventilation_is_on)
        room_to_outside_exchange = self.room_outside_exchange(T_rooms, T_out, ventilation_is_on)
        internal_exchange = self.internal_exchange(T_rooms)

        rhs_rooms = tf.keras.layers.Add()([in_wall_to_room_exchange, room_to_outside_exchange, internal_exchange, u_direct])
        rhs_wall = tf.keras.layers.Add()([in_wall_to_wall_exchange, out_wall_to_wall_exchange])

        T_rooms_new = self.temperature_update_room(T_rooms, rhs_rooms)
        T_wall_new = self.temperature_update_wall(T_wall, rhs_wall)

        return T_rooms_new, T_wall_new
    

def create_tf_single_step_layer(num_rooms: int, delta_t: float, R_inv_internal=None, R_inv_walls=None, C_inv_rooms=None, C_inv_walls=None, u_gains=None):
    print("Creating tf single step layer")
    T_rooms = tf.keras.Input(shape=(num_rooms,1), name="T_rooms")
    T_wall = tf.keras.Input(shape=(num_rooms,1), name="T_wall")

    T_out = tf.keras.Input(shape=(1), name="T_out")
    u_is_on = tf.keras.Input(shape=(1), name="u_is_on")
    ventilation_is_on = tf.keras.Input(shape=(1), name="ventilation_is_on")


    thermo_layer = ThermoPBMLayer(num_rooms, delta_t, R_inv_internal, R_inv_walls, C_inv_rooms, C_inv_walls, u_gains)

    T_new_rooms, T_new_wall = thermo_layer.call(T_rooms, T_wall, T_out, u_is_on, ventilation_is_on)

    return tf.keras.Model(inputs=[T_rooms, T_wall, T_out, u_is_on, ventilation_is_on], outputs=[T_new_rooms, T_new_wall])

if __name__ == '__main__':
    #Asset values
    sim_asset = asset.get_asset()
    
    R_inv_internal = sim_asset.R_partWall_inv
    
    R_inv_wall = sim_asset.get_R_inv()

    C_inv_rooms = sim_asset.C_room_inv
    C_inv_walls = sim_asset.C_wall_inv

    num_rooms = C_inv_rooms.size

    R_inv_wall = np.hstack((R_inv_wall, np.zeros((num_rooms, 1))))


    #Initial values
    T_rooms = np.array([[[3], [3], [3]]])
    T_wall = np.array([[[1], [1], [1]]])
    T_out = np.array([[1]])

    u_gains = np.array([[0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0]])

    delta_t = 1e-2

    tf_single_step_layer = create_tf_single_step_layer(num_rooms, delta_t, R_inv_internal, R_inv_wall, C_inv_rooms, C_inv_walls, u_gains)
    tf_single_step_layer.summary(line_length=150)

    T_new_rooms, T_new_wall = T_rooms, T_wall

    u_is_on = np.array([[0]])
    ventilation_is_on = np.array([[0]])

    T = 1
    for i in range(int(T/delta_t)):
        T_new_rooms, T_new_wall = tf_single_step_layer([T_new_rooms, T_new_wall, T_out, u_is_on, ventilation_is_on])

    print(T_new_rooms)
    print(T_new_wall)