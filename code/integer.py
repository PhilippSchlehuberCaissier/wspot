from dataclasses import dataclass
from semiring import Semiring


## Class representing the ring of integers. This class is only used for test purposes.
@dataclass
class Integer(Semiring):
    @staticmethod
    @property
    def zero():
        return 0

    def __add__(self, other):
        return self + other

    @staticmethod
    @property
    def one():
        return 1

    def __mul__(self, other):
        return self * other

    @staticmethod
    def transition_to_sr(e, weight):
        return weight
