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

    
            
gfBedroom = Zone("gfBedroom",       [0.34146, 1.36584, 1.36584, 0.34146],           [41652.8784, 402850.8])
gfLivingroom = Zone("gfLivingroom", [0.598068, 2.392272, 2.392272, 0.598068],       [157355.3184, 747837.36])
stairs = Zone("stairs",             [0.4067498, 1.6269992, 1.6269992, 0.4067498],   [69035.7892, 647290.4264])
gfBath = Zone("gfBath",             [0.2705976, 1.0823904, 1.0823904, 0.2705976],   [44159.7646, 434698.992])
gfStorage = Zone("gfStorage",       [0.2901384, 1.1605536, 1.1605536, 0.2901384],   [49173.537, 466090.128])
f1Guestroom = Zone("f1Guestroom",   [0.366948, 1.467792, 1.467792, 0.366948],       [49944.8866, 443795.76])
f1Mainroom = Zone("f1Mainroom",     [0.421254, 1.685016, 1.685016, 0.421254],       [66721.7404, 525431.88])
f1Sleep3 = Zone("f1Sleep3",         [0.370791, 1.483164, 1.483164, 0.370791],       [49559.2118, 486390.42])
f1Bath = Zone("f1Bath",             [0.172752, 0.691008, 0.691008, 0.172752],       [37796.1304, 277515.84])
f1Storage = Zone("f1Storage",       [0.423831, 1.695324, 1.695324, 0.423831],       [26033.049, 673890.636])
f1Entrance = Zone("f1Entrance",     [0.123192, 0.492768, 0.492768, 0.123192],       [88512.3666, 197900.64])
f2Livingroom = Zone("f2Livingroom", [2.6243349, 10.4973396, 10.4973396, 2.6243349], [345178.946, 3373189.904])
f2Office = Zone("f2Office",         [0.3805635, 1.522254, 1.522254, 0.3805635],     [42424.228, 462828.5916])
zones = [gfBedroom, gfLivingroom, stairs, gfBath, gfStorage, f1Guestroom, f1Mainroom, f1Sleep3, f1Bath, f1Storage, f1Entrance, f2Livingroom, f2Office]

connections =  [{"rooms": ["gfBedroom", "gfLivingroom"],    "R": 10.35216, "C": 179373.6}, 
                {"rooms": ["gfBedroom", "stairs"],          "R": 4.43664, "C": 76874.4},
                {"rooms": ["gfBedroom", "f1Guestroom"],     "R": 2.016, "C": 207012.6},
                {"rooms": ["gfLivingroom", "gfBath"],       "R": 9.785256, "C": 169550.76},
                {"rooms": ["gfLivingroom", "stairs"],       "R": 38.10888, "C": 0},
                {"rooms": ["gfLivingroom", "f1Entrance"],   "R": 2.7216, "C": 279467.01},
                {"rooms": ["gfLivingroom", "f1Mainroom"],   "R": 3.2256, "C": 331220.16},
                {"rooms": ["gfBath", "gfStorage"],          "R": 5.91552, "C": 102499.2},
                {"rooms": ["gfBath", "f1Sleep3"],           "R": 1.12896, "C": 115927.056},
                {"rooms": ["gfStorage", "stairs"],          "R": 4.43664, "C": 76874.4},
                {"rooms": ["gfStorage", "f1Bath"],          "R": 1.82784, "C": 187691.424},
                {"rooms": ["f1Guestroom", "f1Mainroom"],    "R": 7.3944, "C": 128124},
                {"rooms": ["f1Guestroom", "f1Entrance"],    "R": 4.43664, "C": 76874.4},
                {"rooms": ["f1Guestroom", "stairs"],        "R": 4.43664, "C": 76874.4},
                {"rooms": ["f1Guestroom", "f2Livingroom"],  "R": 2.4192, "C": 248415.12},
                {"rooms": ["f1Mainroom", "f1Sleep3"],       "R": 5.17608, "C": 89686.8},
                {"rooms": ["f1Mainroom", "f1Entrance"],     "R": 6.65496, "C": 115311.6},
                {"rooms": ["f1Mainroom", "f2Livingroom"],   "R": 3.2256, "C": 331220.16},
                {"rooms": ["f1Sleep3", "f1Entrance"],       "R": 12.57048, "C": 217810.8},
                {"rooms": ["f1Sleep3", "f2Livingroom"],     "R": 2.39904, "C": 246344.994},
                {"rooms": ["f1Entrance", "f1Bath"],         "R": 3.94368, "C": 68332.8},
                {"rooms": ["f1Entrance", "stairs"],         "R": 38.10888, "C": 0},
                {"rooms": ["f1Entrance", "f2Livingroom"],   "R": 4.28064, "C": 439556.754},
                {"rooms": ["f1Bath", "f1Storage"],          "R": 0.72216, "C": 116010.72},
                {"rooms": ["f1Bath", "stairs"],             "R": 4.43664, "C": 76874.4},
                {"rooms": ["f1Bath", "f2Office"],           "R": 1.82784, "C": 187691.424},
                {"rooms": ["f2Livingroom", "f2Office"],     "R": 12.089844, "C": 209482.74},
                {"rooms": ["f2Livingroom", "stairs"],       "R": 43.131254, "C": 0},
                {"rooms": ["f2Office", "stairs"],           "R": 4.729335, "C": 81945.975}]
zones = [gfBedroom, gfLivingroom, stairs, gfBath, gfStorage, f1Guestroom, f1Mainroom, f1Sleep3, f1Bath, f1Storage, f1Entrance, f2Livingroom, f2Office]
asset = Asset(zones, connections)

out_temperature = 20
initial_zone_temperature = [3, 3, 3]
initial_wall_temperature = [1, 1, 1]

def get_asset():
    return asset

def get_initial_values():
    return np.array(initial_zone_temperature).reshape((1, len(zones), 1)), np.array(initial_wall_temperature).reshape((1, len(zones), 1)), np.array(out_temperature).reshape((1,1))