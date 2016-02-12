import math

import numpy as np


class Vector(object):
    def __init__(self, coordinates):
        """
        Vector class to do some different types of mathematical functions
        on vector(s). Hacked together in a way that you often need to provide
        2 vectors for a function, but the init only takes 1 vector.
        Can see an example of how to do that in the out-of-class
        functions below. Should/might fix
        :param coordinates:
        :return:
        """
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.dimensions = len(coordinates)
        except ValueError:
            raise ValueError('Coordinates must be nonempty')
        except TypeError:
            raise TypeError('Coordinates must be iterable')

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates

    def get_magnitude(self, vector=None):
        """
        Referred to as ||v||
        Returns the magnitude of a vector
        :param vector: vector used, defaul: None
        :return: magnitude of vector
        """
        if vector is None:
            vector = self.coordinates
        return (sum([x ** 2 for x in vector])) ** .5

    def normalize_vector(self, vector=None):
        """

        :param vector: vector used, defaul: None
        :return: normalization of vector
        """
        if vector is None:
            vector = self.coordinates
        magnitude = self.get_magnitude(vector)
        normalized = [(1 / magnitude) * x for x in vector]
        # assert self._get_magnitude(normalized) == 1.0
        return normalized

    def dot_product(self, v2):
        """
        Return dot product of two vectors
        :param v1:
        :param v2:
        :return:
        """
        return sum([x[0] * x[1] for x in zip(self.coordinates, v2)])

    def get_theta(self, v2, returnable='radians'):
        """
        Theta is the angle between V/W if we drew them starting
            at the same point
        Technically there are two angles between two vectors
            We always use the shorter one
        Using the inverse of the cos function, also called the arccos,
            we can solve for this angle.

        :param v1: Vector coordinates 1
        :param v2: Vector coordinates 2
        :param returnable: radians or degrees
        :return: radians or degrees
        """
        numerator = self.dot_product(self.coordinates, v2)
        den = self.get_magnitude(self.coordinates) * self.get_magnitude(v2)
        radians = math.acos(numerator / den)
        if returnable == 'radians':
            return radians
        else:
            return math.degrees(radians)

    def check_parallel(self, v2):
        """
        vector v1 is parallel to v2 if one is a scalar
        multiple of the other:
        v is parallel to: 2v, .5v, 1v, -v, 0
        If we can multiply one by a scalar to get the other
        So v1 * 2 = v2, parallel!
        :param v1:
        :param v2:
        :return:
        """
        x = np.cross(self.coordinates, v2)
        # np.cross will give us either X,Y,Z (what we have to multiply by)
        # or it will give us one number we can multiply V1 by to get V2
        if x.shape == ():
            return True
        return False

    def check_orthogonal(self, v2, tolerance=1e-10):
        """
        v1 and v2 are orthogonal if their dot product is
        0. That means either v1 or v2 is the 0 vector,
        or they are at right angles
        :param v1:
        :param v2:
        :return:
        """
        dp = self.dot_product(self.coordinates, v2)
        if abs(dp) < tolerance:
            return True
        return False


def mag_and_dir():
    """
    Answers for Lesson 1, Vectors, part about Magnitude and Direction
    :return:
    """
    q1 = Vector([-.221, 7.437])
    print q1.get_magnitude()
    q2 = Vector([8.813, -1.331, -6.247])
    print q2.get_magnitude()
    q3 = Vector([5.581, -2.136])
    print q3.normalize_vector()
    q4 = Vector([1.996, 3.108, -4.554])
    print q4.normalize_vector()


def dot_product():
    """
    Magnitude = ||v|| or mag(v)
    Cauchy-Schwarz Inequality:
        Absolute value of v*w is <= mag(v)*mag(w)
    If v*w = mag(v)*mag(w):
        Means that the cosign of theta = 1, meaning theta
            must equal 0, meaning vectors are pointing in the same
            direction
    If v*w = - mag(v)*mag(w):
        Means the lines are pointing in opposite directions
    If v*w = 0, but v != 0 and w != 0
        Then cos(theta) = 0, theta = pi/2 (90degrees)
        v and w are at a right angle
    v * v = magnitude(v) ** 2
        Angle between both is 0, cosign(theta) = 1
        ||V|| = sqrt(v*v)
    :return:
    """
    v1 = Vector([7.887, 4.138])
    v2 = Vector([-8.802, 6.776])
    q1 = v1.dot_product(v2.coordinates)
    print q1
    v1 = Vector([-5.955, -4.904, -1.874])
    v2 = Vector([-4.496, -8.755, 7.103])
    q2 = v1.dot_product(v2.coordinates)
    print q2


def angles():
    v1 = Vector([3.183, -7.627])
    v2 = Vector([-2.668, 5.319])
    print v1.get_theta(v2.coordinates, returnable='radians')
    v1 = Vector([7.35, .221, 5.188])
    v2 = Vector([2.751, 8.259, 3.985])
    print v1.get_theta(v2.coordinates, returnable='degrees')


def parallel_or_orthogonal():
    v1 = Vector([-7.579, -7.88])
    v2 = Vector([22.737, 23.64])
    v1.check_parallel(v2.coordinates)
    v1.check_orthogonal(v2.coordinates)
    v1 = Vector([-2.029, 9.96, 4.172])
    v2 = Vector([-9.231, -6.639, -7.245])
    v1.check_parallel(v2.coordinates)
    v1.check_orthogonal(v2.coordinates)
    v1 = Vector([-2.328, -7.284, -1.214])
    v2 = Vector([-1.821, 1.072, -2.94])
    v1.check_parallel(v2.coordinates)
    v1.check_orthogonal(v2.coordinates)
    v1 = Vector([2.118, 4.827])
    v2 = Vector([0, 0])
    v1.check_parallel(v2.coordinates)
    v1.check_orthogonal(v2.coordinates)


if __name__ == '__main__':
    # mag_and_dir()
    # dot_product()
    # angles()
    parallel_or_orthogonal()
