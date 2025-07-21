class Semiring:
    ## The neutral element for the additive operator.
    @staticmethod
    def zero():
        raise NotImplementedError

    ## The additive operator.
    def __add__(self, other):
        raise NotImplementedError

    ## The neutral element for the multiplicative operator.
    @staticmethod
    def one():
        raise NotImplementedError

    ## The multiplicative operator.
    def __mul__(self, other):
        raise NotImplementedError

    ## Convert a transition in a weighted automaton into an element of this semiring.
    @staticmethod
    def transition_to_sr(e, weight):
        raise NotImplementedError
