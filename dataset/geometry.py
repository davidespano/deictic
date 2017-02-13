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
