import math

class Geometry2D:

    @staticmethod
    def  distance(px, py, qx, qy):
        dx = qx - px
        dy = qy - py
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def pathLength(sequence, cols=[0,1]):
        length = 0
        for i in range(1,len(sequence)):
            px = sequence[i-1, cols[0]]
            py = sequence[i-1, cols[1]]
            qx = sequence[i, cols[0]]
            qy = sequence[i, cols[1]]
            length += Geometry2D.distance(px, py, qx, qy)
        return length

    @staticmethod
    def boundingBox2D(sequence, cols=[0,1]):
        minx = min(sequence[:,cols[0]])
        miny = min(sequence[:,cols[1]])
        maxx = max(sequence[:, cols[0]])
        maxy = max(sequence[:, cols[1]])
        return minx, miny, maxx - minx, maxy - miny

    @staticmethod
    def getCosineFromSides(x, y):
        r = math.sqrt(x * x + y * y)
        return x / r

    @staticmethod
    def getCosineFromSides(x,y,z):
        rx = [
                [1, 0, 0],
                [0, math.cos(theta[0]), -sin(theta[0])],
                [0, sin]
             ]
    ##########################################################
    # Compute the centroid of the given points. The centroid is defined as the
    # average x and average y values, i.e., (x_bar, y_bar).
    @staticmethod
    def Centroid(sequence, cols=[0,1]):
        xsum = 0;
        ysum = 0;

        #
        for index in range(0, len(sequence)):
            xsum = xsum + sequence[index, cols[0]]
            ysum = ysum + sequence[index, cols[1]]

        a = xsum/len(sequence)
        b = ysum/len(sequence)
        return a, b


class Geometry3D:
    @staticmethod
    def getCosineFromSides(x, y, z):
        r_zyx = r(z,z) * r(y,y) * r(x,x)

        #r = math.sqrt(x * x + y * y)
        return x / r