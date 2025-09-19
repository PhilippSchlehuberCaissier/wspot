from dataclasses import dataclass
from semiring import Semiring


## Class representing the tropical semiring. This class is only used for test purposes.
@dataclass
class Integer(Semiring):
    def __init__(self, what):
        self.what = what

    @staticmethod
    def zero():
        # not max int but big enough
        return Integer(-999999)

    def __add__(self, other):
        return Integer(max(self.what, other.what))
    
    @staticmethod
    def one():
        return Integer(0)

    def __mul__(self, other):
        return Integer(self.what + other.what) if self != Integer.zero() and other != Integer.zero() else Integer.zero()

    def __gt__(self, other):
        return self.what > other.what

    @staticmethod
    def transition_to_sr(e, weight):
        return Integer(weight)
