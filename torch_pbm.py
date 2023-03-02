
import numpy as np
import asset
import torch 
from torch import nn

# TODO: Fix non negative constraint for R, C  and u_gain

def torch_divide_no_nan(x, y, name=None):
    return torch.where(torch.eq(y, 0), torch.zeros_like(y), torch.div(x, y))

class InternalExchangeLayer(nn.Module):
    def __init__(self, R_inv_internal_on, R_inv_internal_off=None, trainable=True, name=None):
        super().__init__()
        num_rooms = R_inv_internal_on.shape[0]
        self.num_rooms = num_rooms
        # TODO: Ensure that  the matrixes are symmetric
        self.R_inv_combination = InputCombinationLayer(R_inv_internal_on, R_inv_internal_off, trainable=trainable, name="R_inv_internal", inverse=True)

    def forward(self, T_rooms, R_is_on):
        T_rooms_tiled = torch.tile(T_rooms, (1,1, self.num_rooms))
        T_rooms_tiled_tran = torch.transpose(T_rooms_tiled, 1, 2)
        skew_diff = T_rooms_tiled_tran - T_rooms_tiled
        R_inv = self.R_inv_combination(R_is_on)
        individual_exchange = R_inv * skew_diff
        summed_exchange = torch.sum(individual_exchange, 2, keepdim=True)
        return summed_exchange

class TemperatureUpdateLayer(nn.Module):
    def __init__(self, C_inv, delta_t, trainable=True, name=None):
        super().__init__()
        self.delta_t = torch.tensor(delta_t)
        C_inv_tensor = torch.tensor(C_inv.reshape((1, *C_inv.shape)))
        self.C_inv = nn.Parameter(C_inv_tensor, requires_grad=trainable)
    
    def forward(self, T, rhs):

        change = self.C_inv * rhs
        T_new = torch.add(T, self.delta_t*change)
        return T_new

class DirectExchangeLayer(nn.Module):
    def __init__(self, R_inv_on, R_inv_off, trainable=True, name=None):
        super().__init__()

        self.R_combination = InputCombinationLayer(R_inv_on, R_inv_off, trainable=trainable, name="R_inv_direct", inverse=True)

    
    def forward(self, T_rooms, T_compare, is_on):
        diff = T_compare - T_rooms
        R_inv = self.R_combination(is_on)
        exchange = R_inv * diff

        return exchange

class WallOutsideExchangeLayer(nn.Module):
    def __init__(self, R_inv_out_wall_outside, R_inv_out_wall, u_outside_on, u_outside_off, trainable=True, u_gains_trainable=True, R_walls_trainable=True, name=None):
        super().__init__()
        self.num_rooms = R_inv_out_wall_outside.shape[0]
        R_inv_out_wall_outside_tensor = torch.tensor(R_inv_out_wall_outside.reshape((1, *R_inv_out_wall_outside.shape)))
        self.R_inv_out_wall_outside = nn.Parameter(R_inv_out_wall_outside_tensor, requires_grad=trainable and R_walls_trainable)
        
        R_inv_out_wall_tensor = torch.tensor(R_inv_out_wall.reshape((1, *R_inv_out_wall.shape)))
        self.R_inv_out_wall = nn.Parameter(R_inv_out_wall_tensor, requires_grad=trainable and R_walls_trainable)

        self.u_outside = InputCombinationLayer(u_outside_on, u_outside_off, trainable=trainable and u_gains_trainable, name="u_outside")
        
        self.divide = torch_divide_no_nan

    def forward(self, T_wall, T_out, u_is_on):

        u_outside = self.u_outside(u_is_on)
        u_outside_sum = torch.sum(u_outside, 2, keepdim=True)

        diff = T_out - T_wall
        R_prod = self.R_inv_out_wall*self.R_inv_out_wall_outside
        R_sum = self.R_inv_out_wall + self.R_inv_out_wall_outside
        scaled_diff = R_prod * diff

        out_wall_to_wall_exchange = self.divide(self.R_inv_out_wall* u_outside_sum + scaled_diff, R_sum)

        return out_wall_to_wall_exchange


class RoomWallExchange(nn.Module):
    def __init__(self, R_inv_in_wall, R_inv_room_wall, u_inside_on, u_inside_off, trainable=True, u_gains_trainable=True, R_walls_trainable=True, name=None):
        super().__init__()
        self.num_rooms = R_inv_in_wall.shape[0]
        self.R_inv_in_wall = nn.Parameter(torch.tensor(R_inv_in_wall.reshape((1, *R_inv_in_wall.shape))), requires_grad=trainable and R_walls_trainable)
        self.R_inv_room_wall = nn.Parameter(torch.tensor(R_inv_room_wall.reshape((1, *R_inv_room_wall.shape))), requires_grad=trainable and R_walls_trainable)

        self.u_inside = InputCombinationLayer(u_inside_on, u_inside_off, trainable=trainable and u_gains_trainable, name="u_inside")
        self.divide = torch_divide_no_nan
    

    def forward(self, T_rooms, T_wall, u_is_on):

        u_inside = self.u_inside(u_is_on)
        u_inside_sum = torch.sum(u_inside,2, keepdim=True)

        in_wall_R_prod_T_diff = self.R_inv_in_wall * self.R_inv_room_wall * (T_rooms - T_wall)
        in_wall_R_sum = self.R_inv_in_wall + self.R_inv_room_wall

        in_wall_to_wall_exchange = self.divide(self.R_inv_in_wall * u_inside_sum + in_wall_R_prod_T_diff, in_wall_R_sum)

        in_wall_to_room_exchange = self.divide(self.R_inv_room_wall * u_inside_sum - in_wall_R_prod_T_diff, in_wall_R_sum)

        return in_wall_to_wall_exchange, in_wall_to_room_exchange

class LinearCombination(nn.Module):
    def __init__(self, on_gain, off_gain, trainable=True):
        super().__init__()
        on_gain = torch.tensor(on_gain.reshape(1, *on_gain.shape))
        off_gain = torch.tensor(off_gain.reshape(1, *off_gain.shape))
        self.on_gain = nn.Parameter(on_gain, requires_grad=trainable)
        self.off_gain = nn.Parameter(off_gain, requires_grad=trainable)
        self.one = torch.tensor(1.0)
    
    def forward(self, x):
        x_opposite = self.one - x
        return torch.add(torch.mul(self.on_gain,x), torch.mul(self.off_gain,x_opposite))

class InverseLinearCombination(nn.Module):
    def __init__(self, on_gain, off_gain, trainable=True):
        super().__init__()
        on_gain = torch.tensor(on_gain.reshape(1, *on_gain.shape))
        off_gain = torch.tensor(off_gain.reshape(1, *off_gain.shape))
        self.on_gain = nn.Parameter(on_gain, requires_grad=trainable)
        self.off_gain = nn.Parameter(off_gain, requires_grad=trainable)
        self.divide = torch_divide_no_nan
    
    def forward(self, x):
        linear = torch.add(self.on_gain* (1-x), self.off_gain * x)
        prod = self.on_gain * self.off_gain
        return self.divide(prod, linear)

class VariableReturn(nn.Module):
    def __init__(self, var, trainable=True):
        super().__init__()
        var = torch.tensor(var.reshape(1, *var.shape))
        self.var = nn.Parameter(var, requires_grad=trainable)
    
    def forward(self, x):
        return self.var
  

class InputCombinationLayer(nn.Module):
    def __init__(self, on_gain, off_gain=None, inverse=False, trainable=True, name=None):
        super().__init__()
        if off_gain is None:
            self.combination = VariableReturn(on_gain, trainable=trainable)
        elif inverse:
            self.combination = InverseLinearCombination(on_gain, off_gain, trainable=trainable)
        else:
            self.combination = LinearCombination(on_gain, off_gain, trainable=trainable)
    
    def forward(self, input_is_on):
        return self.combination(input_is_on)


class ThermoPBMLayer(nn.Module):
    # Num parameters = num_rooms^2 + 14 * num_rooms
    def __init__(self, 
        num_rooms: int,
        delta_t: float,
        num_u = 1,
        num_direct_connections = 1, 

        R_inv_internal=None, 
        variable_R_inv_internal=False,
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
        super().__init__()
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

        self.delta_t = torch.tensor(delta_t)



    
    def forward(self, T_rooms, T_wall, T_out, T_direct_connections, u_is_on, internal_exchange_is_on, direct_connections_is_on):

        u_direct = self.u_direct(u_is_on)
        u_direct_sum = torch.sum(u_direct, 2, keepdim=True)

        in_wall_to_wall_exchange, in_wall_to_room_exchange = self.room_wall_exchange(T_rooms, T_wall, u_is_on)
        out_wall_to_wall_exchange = self.wall_outside_exchange(T_wall, T_out, u_is_on)

        sum_direct_connections_exchange = torch.zeros_like(T_rooms)
        for i in range(self.num_direct_connections):
            direct_exchange = self.room_direct_connections_exchange[i](T_rooms, T_direct_connections[:,i], direct_connections_is_on[:,i])
  
            sum_direct_connections_exchange = torch.add(sum_direct_connections_exchange, direct_exchange)

        internal_exchange = self.internal_exchange(T_rooms, internal_exchange_is_on)

        rhs_rooms = torch.sum(torch.stack([in_wall_to_room_exchange, internal_exchange, u_direct_sum, sum_direct_connections_exchange]), dim=0)
        rhs_wall = torch.add(in_wall_to_wall_exchange, out_wall_to_wall_exchange)

        T_rooms_new = self.temperature_update_room(T_rooms, rhs_rooms)
        T_wall_new = self.temperature_update_wall(T_wall, rhs_wall)

        return T_rooms_new, T_wall_new
    


if __name__ == '__main__':


    
    
    #Asset values
    #sim_asset = asset.get_asset()
    
    #R_inv_internal = sim_asset.R_partWall_inv
    
    #R_inv_wall = sim_asset.get_R_inv()

    #C_inv_rooms = sim_asset.C_room_inv
    #C_inv_walls = sim_asset.C_wall_inv

    #num_rooms = C_inv_rooms.size
    num_rooms = 3
    num_u = 2
    num_ventilation = 2

    #R_inv_wall = np.hstack((R_inv_wall, np.zeros((num_rooms, 1))))

    torch.tensor
    #Initial values
    T_rooms = torch.tensor([[[3], [3], [3]]])
    T_wall = torch.tensor([[[1], [1], [1]]])
    T_out = torch.tensor([[1]])

    T_direct_connections = torch.tensor([[1, 1]])

    u_gains = np.zeros((num_u, num_rooms, 6))
    R_inv_ventilation = np.zeros((num_ventilation, num_rooms, 2))


    delta_t = 1e-2

    single_step_layer = ThermoPBMLayer(num_rooms, delta_t, num_u=2, num_direct_connections=2)

    T_new_rooms, T_new_wall = T_rooms, T_wall

    u_is_on = torch.tensor([[[0, 1],
                         [1, 0],
                         [0, 0]]])

    direct_connection_is_on = torch.tensor([[[[0], [0], [1]],
                                        [[1], [1],[0]]]])

    R_internal_on = torch.tensor([[[0, 1, 0],
                              [1, 0, 0],
                              [0, 0, 0]]])

    T = 1
    for i in range(int(T/delta_t)):
        T_new_rooms, T_new_wall = single_step_layer(T_new_rooms, T_new_wall, T_out, T_direct_connections, u_is_on, R_internal_on, direct_connection_is_on)

    print(T_new_rooms)
    print(T_new_wall)
    