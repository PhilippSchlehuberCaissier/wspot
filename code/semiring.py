class Semiring:
    @staticmethod
    def zero():
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    @staticmethod
    def one():
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    @staticmethod
    def transition_to_sr(e, weight):
        raise NotImplementedError
