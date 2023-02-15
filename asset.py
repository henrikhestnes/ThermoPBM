import numpy as np

class Zone:
    def __init__(self, zone_name: str, R_inv: list[float], C_inv: list[float]):
        """
            zone_name: name of zone:)
            R_inv: 1D array of inverse resistance in the zone in the following order:
                R_ext_inv: Inverse resistance of the thin air layer at the exterior wall exterior surface
                R_outWall_inv: Inverse resistance of the outer part of the exterior wall
                R_inWall_inv: Inverse resistance of the inner part of the exterior wall
                R_room_inv: Inverse resistance of the thin air layer at the exterior wall interior surface
                R_infiltration_inv: Inverse variable resistance to the externalenvironment due to air leakage, windows, cracks, etc.
            C_inv: 1D array of inverse capasitance in the zone in the following order:
                C_wall: Inverse capacitance of the heavy wall material in the room
                C_room: Inverse capacitance of the air and furniture in the room
        """
        self.name = zone_name
        self.R_values_inv = R_inv
        self.C_values_inv = C_inv
    
    def update_infiltration(self, R_infiltration_inv):
        self.R_infiltration_inv = R_infiltration_inv

    def get_R_inv(self):
        return self.R_values_inv
    
    def get_C_inv(self):
        return self.C_values_inv


class Asset:
    def __init__(self, zones: list[Zone], R_partWall_inv: np.ndarray, C_partWall_inv: np.ndarray):
        """
            Zones: list of all zones in asset
            R_partWall_inv: Symmetric 2D array of inverse resistance between zones
            C_partWall_inv: Symmetric 2D array of inverse capacitance between zones 
        """
        self.zones = zones

        self.R_ext_inv = np.zeros(len(zones))
        self.R_outWall_inv = np.zeros(len(zones))
        self.R_inWall_inv = np.zeros(len(zones))
        self.R_room_inv = np.zeros(len(zones))
        self.R_infiltration_inv = np.zeros(len(zones))

        self.C_wall_inv = np.zeros((len(zones), 1))
        self.C_room_inv = np.zeros((len(zones), 1))

        for i, zone in enumerate(zones):
            zone_R_inv = zone.get_R_inv()
            self.R_ext_inv[i] = zone_R_inv[0]
            self.R_outWall_inv[i] = zone_R_inv[1]
            self.R_inWall_inv[i] = zone_R_inv[2]
            self.R_room_inv[i] = zone_R_inv[3]
            self.R_infiltration_inv[i] = zone_R_inv[4]

            zone_C_inv = zone.get_C_inv()
            self.C_wall_inv[i, 0] = zone_C_inv[0]
            self.C_room_inv[i, 0] = zone_C_inv[1]
        
        self.R_partWall_inv = R_partWall_inv

        for i, row in enumerate(C_partWall_inv):
            for elem in row:
                self.C_wall_inv[i, 0] += 0.25*elem
                self.C_room_inv[i, 0] += 0.25*elem
    
    def update_infiltration(self, zone_name, R_infiltration_inv):
        for i, zone in enumerate(self.zones):
            if zone.name == zone_name:
                zone.update_infiltration(R_infiltration_inv)
                self.R_infiltration_inv[i] = R_infiltration_inv
                return
            
    
    def get_R_inv(self):
        return np.array([self.R_ext_inv, self.R_outWall_inv, self.R_inWall_inv, self.R_room_inv, self.R_infiltration_inv]).T

    
            
bedroom = Zone("bedroom", [0,0,10,10,0], [1,1])
livingroom = Zone("livingroom", [0,0.1,10,10,0], [1,1])
bath = Zone("bath", [10,10,1,1,0], [1,1])
zones = [bedroom, livingroom, bath]

R_partWall_inv = np.array([[ 0, 10, 0],
                           [10,  0, 1],
                           [ 0,  1, 0]])

C_partWall_inv = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

asset = Asset(zones, R_partWall_inv, C_partWall_inv)

def get_asset():
    return asset