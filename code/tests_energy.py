import unittest as ut

from integer import Integer
from energy import EnergySegment, EnergyFunction, cross
from semiring import Semiring


WUP = 100
EnergyFunction.set_wup(WUP)

# Some energy segments & functions to play with
f1 = EnergySegment.incr(0, 25, 1, 10)
f2 = EnergySegment.const(0, 25, 2, 20)
f2bis = EnergySegment.const(0, 25, 2, 80)
f3 = EnergySegment.const(5, 40, 3, 50)
f4 = EnergySegment.const(0, 40, 4, 60)
f5 = EnergySegment.const(40, WUP, 5, WUP)

Fnull = EnergyFunction.zero()
Fone = EnergyFunction.one()
F1 = EnergyFunction([f4, f5])
F2 = EnergyFunction([
    EnergySegment.zero(0, 50, None),
    EnergySegment.incr(50, WUP, 1, -50)
])
F3 = EnergyFunction([
    EnergySegment.const(0, 20, 4, 40),
    EnergySegment.incr(20, 80, 5, 20),
    EnergySegment.zero(80, WUP, None)
])


## unittest class for testing semiring-related functions
class TestSemiring(ut.TestCase):
    # Test that every required method is implemented
    def check_impl(self, sr):
        _ = sr.zero()
        _ = sr.one()
        _ = sr.transition_to_sr(None, 0)
        _ = sr.__add__(sr.one(), sr.one())
        _ = sr.__mul__(sr.one(), sr.one())
        return True

    class IncompleteSr(Semiring):
        @staticmethod
        def zero():
            pass

    # Semirings must have an oplus and an otimes with associated neutrals, and must define the associated integer mutator
    def test_incomplete_sr(self):
        self.assertRaises(
            NotImplementedError,
            self.check_impl,
            self.IncompleteSr
        )

    # Example of completely defined semiring
    def test_defined_sr(self):
        self.assertTrue(
            self.check_impl(Integer)
        )



## unittest class for testing the integer semiring
# We use this unittest class to check that __add__ and __mul__ are behaving as intended
class TestInteger(ut.TestCase):
    pass


## unittest class for testing energy functions/segments
class TestEnergy(ut.TestCase):
    ## oplus
    # can't oplus None
    def test_seg_oplus_none(self):
        self.assertRaises(
            TypeError,
            f1.__add__,
            None
        )

    # energy segments on different domains cannot be oplussed
    def test_seg_oplus_no(self):
        self.assertRaises(
            AssertionError,
            f2.__add__,
            f3
        )

    # energy segments on same domains can be oplussed
    def test_seg_oplus_ok(self):
        self.assertEqual(
            f1 + f2bis,
            [f2bis]
        )

    # oplus commutes
    def test_seg_oplus_comm(self):
        self.assertEqual(
            f1 + f2,
            f2 + f1
        )

    # oplus on 2 energy segments with intersection
    def test_seg_oplus_intersect(self):
        self.assertEqual(
            f1 + f2,
            [
                EnergySegment(0, 10, 2, 0, 20),
                EnergySegment(10, 25, 1, 1, 10)
            ]
        )

    ## cross and otimes
    # cross needs an energy function as second argument
    def test_cross_no(self):
        self.assertRaises(
            TypeError,
            cross,
            f1, f2
        )

    # TODO fix this, this is not quite the definition of \bar{1}
    # otimesing with the identity should not change the initial function except for predecessors
    def test_fun_otimes_id(self):
        return
        self.assertEqual(
            F1 * Fone,
            EnergyFunction([
                EnergySegment.const(0, 40, None, 60),
                EnergySegment.const(40, WUP, None, WUP)
            ])
        )

    def test_fun_otimes_id2(self):
        return
        self.assertEqual(
            F2 * Fone,
            EnergyFunction([
                EnergySegment.zero(0, 50, None),
                EnergySegment.incr(50, WUP, None, -50)
            ])
        )

    ## semiring constraints
    # oplus is stable
    def test_fun_oplus_stable(self):
        self.assertTrue(
            isinstance(F1 + F2, EnergyFunction)
        )

    # zero is neutral for oplus
    def test_fun_oplus_zero(self):
        self.assertEqual(
            F1 + EnergyFunction.zero(),
            F1
        )

    # oplus is associative
    def test_fun_oplus_assoc(self):
        self.assertEqual(
            (F1 + F2) + F3,
            F1 + (F2 + F3)
        )

    # oplus is commutative
    def test_fun_oplus_comm(self):
        self.assertEqual(
            F1 + F2,
            F2 + F1
        )

    # otimes is stable
    def test_fun_otimes_stable(self):
        self.assertTrue(
            isinstance(F1 * F2, EnergyFunction)
        )

    # one is neutral for otimes
    def test_fun_otimes_one_left(self):
        self.assertEqual(
            EnergyFunction.one() * F1,
            F1
        )

    def test_fun_otimes_one_right(self):
        self.assertEqual(
            F1 * EnergyFunction.one(),
            F1
        )

    # otimes is associative
    def test_fun_otimes_assoc(self):
        self.assertEqual(
            (F1 * F2) * F3,
            F1 * (F2 * F3)
        )

    # otimes distributes over oplus
    def test_fun_otimes_distrib_left(self):
        self.assertEqual(
            F1 * (F2 + F3),
            (F1 * F2) + (F1 * F3)
        )

    def test_fun_otimes_distrib_right(self):
        self.assertEqual(
            (F2 + F3) * F1,
            (F2 * F1) + (F3 * F1)
        )

    # the null function is an annihilator for otimes
    def test_fun_otimes_null_left(self):
        self.assertEqual(
            F1 * EnergyFunction.zero(),
            EnergyFunction.zero()
        )

    def test_fun_otimes_null_right(self):
        self.assertEqual(
            EnergyFunction.zero() * F1,
            EnergyFunction.zero()
        )


## unittest class for testing solving algos
class TestAlgo(ut.TestCase):
    pass
    

if __name__ == '__main__':
    ut.main()
