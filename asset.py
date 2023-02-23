import numpy as np

class Zone:
    def __init__(self, zone_name: str, R: list[float], C: list[float]):
        """
            zone_name: name of zone:)
            R: 1D array of inverse resistance in the zone in the following order:
                R_ext: Resistance of the thin air layer at the exterior wall exterior surface
                R_outWall: Resistance of the outer part of the exterior wall
                R_inWall: Resistance of the inner part of the exterior wall
                R_room: Resistance of the thin air layer at the exterior wall interior surface
                R_infiltration: Variable resistance to the external environment due to air leakage, windows, cracks, etc.
            C: 1D array of inverse capasitance in the zone in the following order:
                C_wall: Capacitance of the heavy wall material in the room
                C_room: Capacitance of the air and furniture in the room
        """
        self.name = zone_name
        self.R_values = R
        self.C_values = C
    
    def update_infiltration(self, R_infiltration):
        self.R_infiltration = R_infiltration
    
    def get_name(self):
        return self.name

    def get_R(self):
        return self.R_values
    
    def get_C(self):
        return self.C_values


class Asset:
    def __init__(self, zones: list[Zone], connections: list[dict]):
        """
            zones: List of all zones in asset
            connections: Dictionary of all connections in asset
        """
        self.zones = {zone.get_name(): zone for zone in zones}

        self.R_ext = np.zeros(len(zones))
        self.R_outWall = np.zeros(len(zones))
        self.R_inWall = np.zeros(len(zones))
        self.R_room = np.zeros(len(zones))
        self.R_infiltration = np.zeros(len(zones))

        self.C_wall_open = np.zeros((len(zones)))
        self.C_wall_closed = np.zeros((len(zones)))
        self.C_room_open = np.zeros((len(zones)))
        self.C_room_closed = np.zeros((len(zones)))

        for i, zone in enumerate(zones):
            zone_R = zone.get_R()
            self.R_ext[i] = zone_R[0]
            self.R_outWall[i] = zone_R[1]
            self.R_inWall[i] = zone_R[2]
            self.R_room[i] = zone_R[3]
            self.R_infiltration[i] = zone_R[4]

            zone_C = zone.get_C()
            self.C_wall_open[i] = zone_C[0]
            self.C_wall_closed[i] = zone_C[0]
            self.C_room_open[i] = zone_C[1]
            self.C_room_closed[i] = zone_C[1]
        
        self.R_partWall_open = np.full((len(zones), len(zones)), np.inf)
        self.R_partWall_closed = np.full((len(zones), len(zones)), np.inf)
        self.C_partWall_open = np.full((len(zones), len(zones)), np.inf)
        self.C_partWall_closed = np.full((len(zones), len(zones)), np.inf)
        for connection in connections:
            room_1_index = list(self.zones.keys()).index(connection['rooms'][0])
            room_2_index = list(self.zones.keys()).index(connection['rooms'][1])

            self.R_partWall_open[room_1_index, room_2_index] = connection['R_open']
            self.R_partWall_open[room_2_index, room_1_index] = connection['R_open']
            
            self.R_partWall_closed[room_1_index, room_2_index] = connection['R_closed']
            self.R_partWall_closed[room_2_index, room_1_index] = connection['R_closed']
            
            self.C_partWall_open[room_1_index, room_2_index] = connection['C_open']
            self.C_partWall_open[room_2_index, room_1_index] = connection['C_open']
            
            self.C_partWall_closed[room_1_index, room_2_index] = connection['C_closed']
            self.C_partWall_closed[room_2_index, room_1_index] = connection['C_closed']

            
        for row in range(len(zones)):
            for col in range(len(zones)):
                self.C_wall_open[row] += 0.25*self.C_partWall_open[row, col] if self.C_partWall_open[row, col] != np.inf else 0
                self.C_wall_closed[row] += 0.25*self.C_partWall_closed[row, col] if self.C_partWall_closed[row, col] != np.inf else 0
                self.C_room_open[row] += 0.25*self.C_partWall_open[row, col] if self.C_partWall_open[row, col] != np.inf else 0
                self.C_room_closed[row] += 0.25*self.C_partWall_closed[row, col] if self.C_partWall_closed[row, col]!= np.inf else 0
    
    def update_infiltration(self, zone_name, R_infiltration):
        for i, zone in enumerate(self.zones):
            if zone.name == zone_name:
                zone.update_infiltration(R_infiltration)
                self.R_infiltration[i] = R_infiltration
                return

    def get_R_inv(self):
        return np.array([1/self.R_ext, 1/self.R_outWall, 1/self.R_inWall, 1/self.R_room, 1/self.R_infiltration]).T

    def get_C_open_inv(self):
        return np.array([1/self.C_wall_open.reshape(len(zones), 1), 1/self.C_room_open.reshape(len(zones), 1)])    
    
    def get_C_closed_inv(self):
        return np.array([1/self.C_wall_closed.reshape(len(zones), 1), 1/self.C_room_closed.reshape(len(zones), 1)])
    
    def get_R_partWall_open_inv(self):
        return 1/self.R_partWall_open
    
    def get_R_partWall_closed_inv(self):
        return 1/self.R_partWall_closed

    
            
bedroom = Zone("bedroom", [1,1,1,1,np.inf], [1,1])
livingroom = Zone("livingroom", [1,1,1,1,np.inf], [1,1])
bath = Zone("bath", [1,1,1,1,np.inf], [1,1])

connections = [{"rooms": ["bedroom", "livingroom"], "R_open": 1, "R_closed": 10, "C_open": 10, "C_closed": 1},
               {"rooms": ["livingroom", "bath"], "R_open": 1, "R_closed": 10, "C_open": 10, "C_closed": 1}]

zones = [bedroom, livingroom, bath]
asset = Asset(zones, connections)

out_temperature = 20
initial_zone_temperature = [3, 3, 3]
initial_wall_temperature = [1, 1, 1]

def get_asset():
    return asset

def get_initial_values():
    return np.array(initial_zone_temperature).reshape((1, len(zones), 1)), np.array(initial_wall_temperature).reshape((1, len(zones), 1)), np.array(out_temperature).reshape((1,1))