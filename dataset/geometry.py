import math


class Geometry2D:

    @staticmethod
    def distance(px, py, qx, qy):
        dx = qx - px
        dy = qy - py
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def pathLength(sequence, cols=[0, 1]):
        length = 0
        for i in range(1, len(sequence)):
            px = sequence[i - 1, cols[0]]
            py = sequence[i - 1, cols[1]]
            qx = sequence[i, cols[0]]
            qy = sequence[i, cols[1]]
            length += Geometry2D.distance(px, py, qx, qy)
        return length

    @staticmethod
    def boundingBox2D(sequence, cols=[0, 1]):
        minx = min(sequence[:, cols[0]])
        miny = min(sequence[:, cols[1]])
        maxx = max(sequence[:, cols[0]])
        maxy = max(sequence[:, cols[1]])
        return minx, miny, maxx - minx, maxy - miny

    @staticmethod
    def getCosineFromSides(x, y):
        r = math.sqrt(x * x + y * y)
        return x / r

    # @staticmethod
    # def getCosineFromSides(x,y,z):
    #     rx = [
    #             [1, 0, 0],
    #             [0, math.cos(theta[0]), -sin(theta[0])],
    #             [0, sin]
    #          ]
    ##########################################################
    # Compute the centroid of the given points. The centroid is defined as the
    # average x and average y values, i.e., (x_bar, y_bar).
    @staticmethod
    def Centroid(sequence, cols=[0, 1]):
        xsum = 0;
        ysum = 0;

        #
        for index in range(0, len(sequence)):
            xsum = xsum + sequence[index, cols[0]]
            ysum = ysum + sequence[index, cols[1]]

        a = xsum / len(sequence)
        b = ysum / len(sequence)
        return a, b

    @staticmethod
    def Collinear(p, q0, q1, eps):
        d = Point2D(q1.x - q0.x, q1.y - q0.y)
        a = Point2D(p.x - q0.x, p.y - q0.y)
        nda = d.x * a.y - d.y * a.x
        ndn = d.x * d.x + d.y * d.y
        ada = a.x * a.x + a.y * a.y

        if nda * nda > eps * ndn * ada:
            return False
        else:
            return True
        #dda = d.dot(a)
        #if dda < -eps * ndn or dda > (1 + eps) * ndn :
        #    return True
        #else:
        #    return False


class Point2D:

    def __init__(self, x=0, y=0):
        self.x = x;
        self.y = y;

    def dot(self, p2):
        return self.x * p2.x + self.y * p2.y


class Geometry3D:

    @staticmethod
    def distance(px, py, pz, qx, qy, qz):
        """

        :param px:
        :param py:
        :param pz:
        :param qx:
        :param qy:
        :param qz:
        :return:
        """
        dx = qx - px
        dy = qy - py
        dz = qz - pz
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def getCosineFromSides(x, y, z):
        """

        :param x:
        :param y:
        :param z:
        :return:
        """
        r_xy = math.sqrt(x * x + y * y)
        r_zx = math.sqrt(x * x + z * z)
        r_yz = math.sqrt(y * y + z * z)

        return [math.acos(z / r_zx), math.acos(y / r_yz), math.acos(x / r_xy)]
