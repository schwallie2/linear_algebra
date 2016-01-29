import math


class Vector(object):
    def __init__(self, coordinates):
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
        Referred to as || ||
        :param vector:
        :return:
        """
        if vector is None:
            vector = self.coordinates
        return (sum([x ** 2 for x in vector]))**.5

    def get_direction(self):
        pass

    def normalize_vector(self, vector=None):
        if vector is None:
            vector = self.coordinates
        magnitude = self.get_magnitude(vector)
        normalized = [(1 / magnitude) * x for x in vector]
        print 'Normalized: %s' % self.get_magnitude(normalized)
        # assert self._get_magnitude(normalized) == 1.0
        return normalized

    def dot_product(self, v1, v2):
        return sum([x[0] * x[1] for x in zip(v1, v2)])

    def get_theta(self, v1, v2, returnable='radians'):
        """
        Theta is the angle between V/W if we drew them starting
            at the same point
        Technically there are two angles between two vectors
            We always use the shorter one
        Using the inverse of the cos function, also called the arccos,
            we can solve for this angle.
        arccos(normalized(v1)*normalized(v2))

        :return: radians
        """
        numerator = self.dot_product(v1, v2)
        den = self.get_magnitude(v1) * self.get_magnitude(v2)
        radians = math.acos(numerator / den)
        if returnable == 'radians':
            return radians
        else:
            return math.degrees(radians)


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
    q1 = v1.dot_product(v1.coordinates, v2.coordinates)
    print q1
    v1 = Vector([-5.955, -4.904, -1.874])
    v2 = Vector([-4.496, -8.755, 7.103])
    q2 = v1.dot_product(v1.coordinates, v2.coordinates)
    print q2

def angles():
    v1 = Vector([3.183, -7.627])
    v2 = Vector([-2.668, 5.319])
    print v1.get_theta(v1.coordinates, v2.coordinates, returnable='radians')
    v1 = Vector([7.35, .221, 5.188])
    v2 = Vector([2.751, 8.259, 3.985])
    print v1.get_theta(v1.coordinates, v2.coordinates, returnable='degrees')

if __name__ == '__main__':
    # mag_and_dir()
    # dot_product()
    angles()
