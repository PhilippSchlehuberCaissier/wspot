from typing import List
from dataclasses import dataclass

from semiring import Semiring

# We use a global wup
@dataclass
class WUP:
    value = 10


def set_wup(wup):
    WUP.value = wup


@dataclass
## Class for representing energy functions segments.
class EnergySegment:
    # The segment on which this function is defined, which is included in [0, wup]
    lowerBound: int
    upperBound: int

    # Get the optimal predecessor on longer paths
    pred: int

    # We can demonstrate that in our case the equation of this segment will always be of the form e_out = a * e_in + b where a is either 0 or 1
    a: int
    b: int = -1

    @property
    def domain(self):
        return range(self.lowerBound, self.upperBound)

    @property
    def image(self):
        if self.a == 0:
            return range(self.b, self.b+1)
        else:
            # It is guaranteed that the image of an energy segment is in [0,wup]
            return range(
                self.lowerBound + self.b,
                self.upperBound + self.b + 1
                )

    def __str__(self):
        str_def = f"[{self.lowerBound}, {self.upperBound}] -> IR"
        str_a = 'e_in' if self.a else ''
        str_b = '' if self.a and not self.b else str(self.b) if not self.a else f" + {self.b}"
        return f"{str_def} ; e_in |---> {str_a}{str_b}" + f" (from {self.pred})"

    def is_in_domain(self, x):
        if x not in range(self.lowerBound, self.upperBound + 1):
            raise ValueError(f"x = {x} is out of the domain of the segment: {self}")

    def evaluate(self, e_in):
        # TODO This works while our weights are integers
        self.is_in_domain(e_in)
        return self.a * e_in + self.b

    @property
    def is_above_one(self):
        return self.evaluate(self.lowerBound) >= self.lowerBound and self.pred is not None

    def restriction(self, low, upp):
        return EnergySegment(low, upp, self.pred, self.a, self.b)

    @staticmethod
    def zero(low, upp, pred):
        return EnergySegment(low, upp, pred, 0, -1)

    @property
    def is_zero(self):
        return self == EnergySegment.zero(self.lowerBound, self.upperBound, self.pred)

    @staticmethod
    def one(low, upp, pred):
        return EnergySegment(low, upp, pred, 1, 0)

    @staticmethod
    def const(low, upp, pred, k):
        return EnergySegment(low, upp, pred, 0, k)

    @staticmethod
    def incr(low, upp, pred, b):
        return EnergySegment(low, upp, pred, 1, b)


@dataclass
## Class for representing an energy function.
class EnergyFunction(Semiring):
    segments: List[EnergySegment]

    def __str__(self):
        return " U ".join([str(seg) for seg in self.segments])

    @staticmethod
    def one(low, upp):
        return EnergyFunction([EnergySegment.one(low, upp, None)])

    @staticmethod
    def zero(low, upp):
        return EnergyFunction([EnergySegment.zero(low, upp, None)])

    @property
    def is_zero(self):
        # We can evaluate the segment in 0 since "normal" segments cannot get below 0
        return len(self.segments) == 1 and self.segments[0].evaluate(0) == -1

    @property
    def domain(self):
        # Assuming segments are ordered
        return range(self.segments[0].lowerBound,
                     self.segments[-1].upperBound + 1)

    @property
    def discontinuities(self):
        # Assuming segments are ordered
        # IMPORTANT: Global lower and upper bounds are also considered as discontinuities!
        final = [seg.lowerBound for seg in self.segments] + [self.segments[-1].upperBound]
        final.sort()
        return final

    def is_in_domain(self, x):
        if x not in self.domain:
            raise ValueError(f"x = {x} is out of the domain of the function: {self}")

    ## Return the segment that is used when evaluating this function at x.
    def get_segment(self, x):
        self.is_in_domain(x)
        
        for seg in self.segments:
            # We assume that if there is a discontinuity at x, the used segment will be the one that has maximal energy.
            # This means that a segment is only usable on [lowerBound, upperBound-1] unless it is the last segment (since there is no next segment to use)
            corrected_upper = seg.upperBound if seg != self.segments[-1] else seg.upperBound + 1
            if x in range(seg.lowerBound, corrected_upper):
                return seg

    def evaluate(self, x):
        self.is_in_domain(x)

        return self.get_segment(x).evaluate(x)

    @staticmethod
    def clean(f):
        new_segs = []
        next_seg = None
        # Whether we need to make another pass
        to_clean = False

        # f must be defined on [0, wup]
        if f.segments[0].lowerBound > 0:
            f.segments.insert(0, EnergySegment.zero(0,
                                                    f.segments[0].lowerBound,
                                                    None))
        if f.segments[-1].upperBound < WUP.value:
            f.segments.append(EnergySegment.zero(f.segments[-1].upperBound,
                                                 WUP.value,
                                                 None))

        for old_seg in f.segments:
            # Cap every segment to wup
            if old_seg.evaluate(old_seg.upperBound) > WUP.value:
                print(f"overflow for segment {old_seg}")
                if next_seg is not None:
                    new_segs.append(next_seg)
                    next_seg = None
                to_clean = True

                if old_seg.a == 0:
                    new_segs.append(EnergySegment.const(old_seg.lowerBound,
                                                        old_seg.upperBound,
                                                        old_seg.pred,
                                                        WUP.value
                                                        ))
                else:
                    disc = int((WUP.value - old_seg.b) / old_seg.a)
                    new_segs.append(EnergySegment.incr(old_seg.lowerBound,
                                                       disc,
                                                       old_seg.pred,
                                                       old_seg.b
                                                       ))
                    new_segs.append(EnergySegment.const(disc,
                                                        old_seg.upperBound,
                                                        old_seg.pred,
                                                        WUP.value
                                                        ))
                continue

            # Same procedure: nullify non-null segments below zero
            if old_seg.evaluate(old_seg.lowerBound) < 0 and not old_seg.is_zero:
                print(f"underflow for segment {old_seg}")
                if next_seg is not None:
                    new_segs.append(next_seg)
                    next_seg = None
                to_clean = True

                if old_seg.a == 0:
                    new_segs.append(EnergySegment.zero(old_seg.lowerBound,
                                                       old_seg.upperBound,
                                                       old_seg.pred
                                                       ))
                else:
                    disc = int(-old_seg.b / old_seg.a)
                    new_segs.append(EnergySegment.zero(old_seg.lowerBound,
                                                       disc,
                                                       old_seg.pred,
                                                       ))
                    new_segs.append(EnergySegment.incr(disc,
                                                       old_seg.upperBound,
                                                       old_seg.pred,
                                                       old_seg.b
                                                       ))
                continue

            if next_seg is None:
                next_seg = old_seg
                continue

            # Remove zero-length segments EXCEPT if they are defined for wup
            if old_seg.lowerBound == old_seg.upperBound and old_seg.lowerBound == WUP.value:
                continue

            # Merge segments with the same equation
            if old_seg.a == next_seg.a and old_seg.b == next_seg.b and old_seg.pred == next_seg.pred:
                next_seg.upperBound = old_seg.upperBound
                # print(f"merging segments, new segment: {next_seg}")
            else:
                # print(f"next segment: {next_seg}")
                new_segs.append(next_seg)
                next_seg = old_seg
        if next_seg is not None:
            new_segs.append(next_seg)
        f = EnergyFunction(new_segs)
        return EnergyFunction.clean(f) if to_clean else f

    @staticmethod
    def max(f1, f2):
        print(f"comparing:\n{f1}\n{f2}")

        new_segs = []
        # The discontinuities of the max are the union of those of the 2 functions
        discs = list(set(f1.discontinuities + f2.discontinuities))
        discs.sort()
        print(f"new f is discontinuous at {discs}")
        for i in range(len(discs) - 1):
            lower = discs[i]
            upper = discs[i+1]

            seg1 = f1.get_segment(lower)
            seg2 = f2.get_segment(lower)

            if seg1.a == seg2.a:
                if seg1.b == seg2.b:
                    next_seg = seg1 if seg2.pred is None else seg2
                else:
                    next_seg = seg1 if seg1.b > seg2.b else seg2
                new_segs.append(next_seg.restriction(lower, upper))
            else:
                # Segments might be intersecting
                # Use the IVT (xor)
                if (seg1.evaluate(lower) > seg2.evaluate(lower)) != (seg1.evaluate(upper) > seg2.evaluate(upper)):
                    # found by manipulating the functions' equations
                    # again, this only works if our weights are ints
                    # intersect is guaranteed to be in ]lower, upper[ with the initial condition
                    intersect = int((seg2.b - seg1.b) / (seg1.a - seg2.a))
                    first_on_top = seg1 if seg1.evaluate(lower) > seg2.evaluate(lower) else seg2
                    new_segs.append(first_on_top.restriction(lower, intersect))
                    second_on_top = seg2 if first_on_top == seg1 else seg1
                    new_segs.append(second_on_top.restriction(intersect, upper))
                else:
                    # Segments do not intersect, we use upper because its image is guaranteed to be not equal (unless seg1 == seg2)
                    next_seg = seg1 if seg1.evaluate(upper) > seg2.evaluate(upper) else seg2
                    new_segs.append(next_seg.restriction(lower, upper))

        f = EnergyFunction(new_segs)
        print(f"the max is {EnergyFunction.clean(f)}")
        return EnergyFunction.clean(f)

    @property
    def is_above_one(self):
        for seg in self.segments:
            if seg.is_above_one:
                return True
        return False

    def __add__(self, other):
        # TODO clean this!!
        print(f"composing {self} with {other}")
        new_segs = []

        if self.is_zero or other.is_zero:
            return EnergyFunction.zero(0, WUP.value)

        for seg in self.segments:
            if seg.is_zero:
                new_segs.append(seg)
                continue

            lower = seg.lowerBound
            upper = seg.upperBound
            im_lower = seg.evaluate(lower)
            im_upper = seg.evaluate(upper)

            print(f"Image of the first segment is [{im_lower}, {im_upper}]")

            # Case f is constant
            if im_lower == im_upper:
                next_seg = EnergySegment.const(lower,
                                               upper,
                                               seg.pred,
                                               other.evaluate(im_lower))
                print(f"next constant seg from {seg} is {next_seg}")
                new_segs.append(next_seg)
                continue

            # Case f is ascending
            # For all segments of g, if the segment intersects with the image of f then apply this segment to the relevant part of the image
            for other_seg in other.segments:
                other_lower = other_seg.lowerBound
                other_upper = other_seg.upperBound
                if other_lower < im_upper and other_upper > im_lower:
                    # We need to stay in the image
                    corr_lower = max(im_lower, other_lower)
                    corr_upper = min(im_upper, other_upper)

                    # Find the inverse by f
                    inv_lower = int((corr_lower - seg.b) / seg.a)
                    inv_upper = int((corr_upper - seg.b) / seg.a)

                    new_a = seg.a * other_seg.a
                    new_b = other_seg.a * seg.b + other_seg.b
                    next_seg = EnergySegment.incr(inv_lower, inv_upper, seg.pred, new_b) if new_a == 1 else EnergySegment.const(inv_lower, inv_upper, seg.pred, new_b)
                    print(f"next seg from {seg} and {other_seg} is {next_seg}")
                    new_segs.append(next_seg)

        f = EnergyFunction(new_segs)
        print(f"{self} + {other} = {EnergyFunction.clean(f)}")
        return EnergyFunction.clean(f)
