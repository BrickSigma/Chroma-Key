import numpy as np
from numba import int64, float32, uint8
from numba.experimental import jitclass
import math

spec = [
    ('col_len', int64),             # a simple scalar field
    ('xs', uint8[:]),               # an array field
    ('ys', uint8[:]),               # an array field
    ('zs', uint8[:]),               # an array field
    ('equations_xy', float32[:,:]),   # an array field
    ('equations_zy', float32[:,:]),   # an array field
    ('offset', float32),            # scalar field
]

@jitclass(spec)
class Analysis:
    def __init__(self, colors: uint8[:,:]):
        self.col_len = len(colors)  # Number of colors present for analysis

        # Individual arrays will hold the blue, green, red values respectively
        self.xs = colors[:,0]
        self.ys = colors[:,1]
        self.zs = colors[:,2]

        self.equations_xy = np.zeros((4,2), dtype=float32)  # List that will contain the first 4 equations
        self.equations_zy = np.zeros((4,2), dtype=float32)  # List that will contain the remaining 4 equations
        
        self.offset = 0.0

    def calc_offset(self):
        """
        This calculates the offset value for the equations.
        This value is calculated by getting the standard deviation of all the red, green, and blue values,
        And halfs it. This offset is used to counter changes in light levels in the background.
        """
        
        sigma_x = np.sum(self.xs) + np.sum(self.ys) + np.sum(self.zs)
        sigma_x_sqrd = 0
        for i in range(len(self.xs)):
            sigma_x_sqrd += self.xs[i]**2 + self.ys[i]**2 + self.zs[i]**2

        n = len(self.xs)*3
        sd = math.sqrt((sigma_x_sqrd/n)-(sigma_x/n)**2)
        self.offset = float32(sd/2)

        print("Offset:", self.offset)
        
    def split_coord(self, axis_x, axis_y):
        min_x, min_y = min(axis_x), min(axis_y)  # Smallest color values
        max_x, max_y = max(axis_x), max(axis_y)  # Largest color values

        M = float32((max_y-min_y)/(max_x-min_x))  # Average gradient of colors
        C = float32(min_y - (M*min_x))  # Y-intercept of color values
        # The equation of the central line of all points if equivilent to y = Mx + C.
        
        M_IN = float32(-1*(1/M))  # Gradient of perpendicular color values from the central line.
        
        a_len, b_len = 0, 0  # This will hold the domain range of color values
        a_coord, b_coord = (0, 0), (0, 0)  # These are the maximum and minimum points above/bellow the central line.

        # The following lists store the points that are above and bellow the central line separately.
        above_line_x = []
        above_line_y = []
        bellow_line_x = []
        bellow_line_y = []
        
        # This separates the points by determining whether they are above the central line or bellow it.
        for i in range(len(axis_x)):
            if (M*axis_x[i]) + C < axis_y[i]:
                x, y = axis_x[i], axis_y[i]
                above_line_x.append(x)
                above_line_y.append(y)
            elif ((M*axis_x[i]) + C) > axis_y[i]:
                x, y = axis_x[i], axis_y[i]
                bellow_line_x.append(x)
                bellow_line_y.append(y)

        # This loop will return the point above the central line that has the largest perpendicular distance from the central line.
        for i in range(len(above_line_x)):
            y_intercept = above_line_y[i] - (M_IN*above_line_x[i])
            x = (y_intercept-C)/(M-M_IN)
            y = (M*x)+C
            p_len = (((x-above_line_x[i])**2)+((y-above_line_y[i]))**2)**0.5
            if p_len > a_len:
                a_len = p_len
                a_coord = (above_line_x[i], above_line_y[i])

        # This loop will return the point bellow the central line that has the largest perpendicular distance from the central line.
        for i in range(len(bellow_line_x)):
            y_intercept = bellow_line_y[i] - (M_IN*bellow_line_x[i])
            x = (y_intercept-C)/(M-M_IN)
            y = (M*x)+C
            p_len = (((x-bellow_line_x[i])**2)+((y-bellow_line_y[i]))**2)**0.5
            if p_len > b_len:
                b_len = p_len
                b_coord = (bellow_line_x[i], bellow_line_y[i])
                
        self.calc_offset()  # Calculate the offset value. This value is added/subtracted to the equation's Y-intercept.

        c1 = float32(max_y - (M_IN*max_x) + self.offset) # Equation 1's Y-Intercept
        c2 = float32(min_y - (M_IN*min_x) - self.offset) # Equation 2's Y-Intercept
        c3 = float32(a_coord[1] - (M*a_coord[0]) + self.offset) # Equation 3's Y-Intercept
        c4 = float32(b_coord[1] - (M*b_coord[0]) - self.offset) # Equation 4's Y-Intercept

        # Return the first set of equations. (4 equation in total)
        return np.array([[M_IN, c1], [M_IN, c2], [M, c3], [M, c4]], dtype=float32)

    # This function generates the eight equations
    def range_format(self):
        self.equations_xy = self.split_coord(self.xs, self.ys)  # Pass the blue and green color values first as x,y coordinates
        self.equations_zy = self.split_coord(self.zs, self.ys)  # Then pass the red and green color values as the z,y coordinates

    # This function is used to see if the color value (b,g,r) is withing the range of color values, using the equations generated. This is one of the slowest parts of the code
    def check_color(self, color):
        if ((self.equations_xy[0,0]*color[0])+self.equations_xy[0,1] >= color[1]) and ((self.equations_xy[1,0]*color[0])+self.equations_xy[1,1] <= color[1]) and ((self.equations_xy[2,0]*color[0])+self.equations_xy[2,1] >= color[1]) and ((self.equations_xy[3,0]*color[0])+self.equations_xy[3,1] <= color[1]):
            if ((self.equations_zy[0,0]*color[2])+self.equations_zy[0,1] >= color[1]) and ((self.equations_zy[1,0]*color[2])+self.equations_zy[1,1] <= color[1]) and ((self.equations_zy[2,0]*color[2])+self.equations_zy[2,1] >= color[1]) and ((self.equations_zy[3,0]*color[2])+self.equations_zy[3,1] <= color[1]):
                return True
            else:
                return False
        else:
            return False

    def get_rules(self):
        return self.equations_xy, self.equations_zy
