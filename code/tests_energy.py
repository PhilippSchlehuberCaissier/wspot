import unittest as ut

from integer import Integer
from energy import EnergySegment, EnergyFunction, cross


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


class TestSemiring(ut.TestCase):
    def test_f(self):
        pass


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

    # otimesing with the identity should not change the initial function
    def test_fun_otimes_id(self):
        self.assertEqual(
            F1 * Fone,
            EnergyFunction([
                EnergySegment.const(0, 40, None, 60),
                EnergySegment.const(40, WUP, None, WUP)
            ])
        )

    def test_fun_otimes_id2(self):
        self.assertEqual(
            F2 * Fone,
            EnergyFunction([
                EnergySegment.zero(0, 50, None),
                EnergySegment.incr(50, WUP, None, -50)
            ])
        )


if __name__ == '__main__':
    ut.main()
