"""
This is a class I built to calculate and determine the range of color values to exclude from the video.
It uses the basic principle of coordinate geometry, generating a total of eight equations used for comparison in the range_format function.
It may be difficult to understand, but it works well.
"""

import numpy as np
from numba import jit, int64, float32, uint8
from numba.experimental import jitclass
import typing, numba

spec = [
    ('col_len', int64),             # a simple scalar field
    ('xs', uint8[:]),               # an array field
    ('ys', uint8[:]),               # an array field
    ('zs', uint8[:]),               # an array field
    ('equations_xy', float32[:,:]),   # an array field
    ('equations_zy', float32[:,:]),   # an array field
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


    def split_coord(self, axis_x, axis_y):
        min_x, min_y = min(axis_x), min(axis_y)  # Smallest color values
        max_x, max_y = max(axis_x), max(axis_y)  # Largest color values

        M = float32((max_y-min_y)/(max_x-min_x))  # Average gradient of colors
        C = float32(min_y - (M*min_x))  # Y-intercept of color values
        M_IN = float32(-1*(1/M))  # Gradient of perpendicular color values from the average
        a_len, b_len = 0, 0
        a_coord, b_coord = (0, 0), (0, 0)

        above_line_x = []
        above_line_y = []
        bellow_line_x = []
        bellow_line_y = []

        for i in range(len(axis_x)):
            if (M*axis_x[i]) + C < axis_y[i]:
                x, y = axis_x[i], axis_y[i]
                above_line_x.append(x)
                above_line_y.append(y)
            elif ((M*axis_x[i]) + C) > axis_y[i]:
                x, y = axis_x[i], axis_y[i]
                bellow_line_x.append(x)
                bellow_line_y.append(y)

        for i in range(len(above_line_x)):
            y_intercept = above_line_y[i] - (M_IN*above_line_x[i])
            x = (y_intercept-C)/(M-M_IN)
            y = (M*x)+C
            p_len = (((x-above_line_x[i])**2)+((y-above_line_y[i]))**2)**0.5
            if p_len > a_len:
                a_len = p_len
                a_coord = (above_line_x[i], above_line_y[i])

        for i in range(len(bellow_line_x)):
            y_intercept = bellow_line_y[i] - (M_IN*bellow_line_x[i])
            x = (y_intercept-C)/(M-M_IN)
            y = (M*x)+C
            p_len = (((x-bellow_line_x[i])**2)+((y-bellow_line_y[i]))**2)**0.5
            if p_len > b_len:
                b_len = p_len
                b_coord = (bellow_line_x[i], bellow_line_y[i])

        c1 = float32(max_y - (M_IN*max_x) + 15) # Equation 1
        c2 = float32(min_y - (M_IN*min_x) - 15) # Equation 2
        c3 = float32(a_coord[1] - (M*a_coord[0]) + 15) # Equation 3
        c4 = float32(b_coord[1] - (M*b_coord[0]) - 15) # Equation 4

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
