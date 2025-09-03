from typing import List, override, ClassVar
from dataclasses import dataclass

from semiring import Semiring


@dataclass
## Class for representing energy functions segments.
class EnergySegment:
    # The segment on which this function is defined, which is included in [0, wup]
    lowerBound: int
    upperBound: int

    # Get the optimal predecessor on longer paths
    # TODO special value (not int) to represent undef instead of None
    pred: int

    # We can demonstrate that in our case the equation of this segment will always be of the form e_out = a * e_in + b where a is either 0 or 1
    a: int
    b: int = -1

    @property
    def domain(self):
        return (self.lowerBound, self.upperBound)

    def __str__(self):
        str_def = f"[{self.lowerBound}, {self.upperBound}] -> IR"
        str_a = 'e_in' if self.a else ''
        str_b = '' if self.a and not self.b else str(self.b) if not self.a else f" + {self.b}"
        return f"{str_def} ; e_in |---> {str_a}{str_b}" + f" (from {self.pred})"

    # TODO energy domains class
    def is_in_domain(self, x):
        return x >= self.lowerBound and x <= self.upperBound

    @property
    def image(self):
        return (self.a * self.lowerBound + self.b,
                self.a * self.upperBound + self.b)

    def evaluate(self, e_in):
        assert self.is_in_domain(e_in)
        return self.a * e_in + self.b

    @property
    def is_above_one(self):
        return self.evaluate(self.lowerBound) >= self.lowerBound and self.pred is not None

    def restriction(self, low, upp):
        if self.lowerBound >= low and self.upperBound <= upp:
            return self
        return EnergySegment(max(self.lowerBound, low),
                             min(upp, self.upperBound),
                             self.pred, self.a, self.b)

    @staticmethod
    def zero(low, upp, pred):
        return EnergySegment(low, upp, pred, 0, -1)

    @property
    def is_zero(self):
        return self.a == 0 and self.b == -1

    @staticmethod
    def one(low, upp, pred):
        return EnergySegment(low, upp, pred, 1, 0)

    @staticmethod
    def const(low, upp, pred, k):
        return EnergySegment(low, upp, pred, 0, k)

    @staticmethod
    def incr(low, upp, pred, b):
        return EnergySegment(low, upp, pred, 1, b)

    def __add__(self, other):
        # Restrictions should be done *before* using the add operator
        if not isinstance(other, EnergySegment):
            raise TypeError("Can only add energy segments to other energy segments")

        assert self.lowerBound == other.lowerBound and self.upperBound == other.upperBound
        segs = []
        if self.a == other.a:
            if self.b == other.b:
                segs = [other] if self.pred is None else [self]
            else:
                segs = [other] if self.b < other.b else [self]
        else:
            # Use the IVT (xor)
            lower = self.lowerBound
            upper = self.upperBound
            if (self.evaluate(lower) > other.evaluate(lower)) != (self.evaluate(upper) > other.evaluate(upper)):
                # TODO compatibility with real numbers? (in intersect)
                # intersect is guaranteed to be in ]lower, upper[ with the initial condition
                intersect = int((other.b - self.b) / (self.a - other.a))
                first_on_top = self if self.evaluate(lower) > other.evaluate(lower) else other
                segs.append(first_on_top.restriction(lower, intersect))
                second_on_top = other if first_on_top == self else self
                segs.append(second_on_top.restriction(intersect, upper))
            else:
                segs = [other] if self.evaluate(upper) < other.evaluate(upper) else [self]
        return segs

        f = EnergyFunction(segs)
        return EnergyFunction.clean(f)


@dataclass
## Class for representing an energy function.
class EnergyFunction(Semiring):
    ZERO: ClassVar
    ONE: ClassVar
    WUP: ClassVar[int]

    segments: List[EnergySegment]
    length: int

    @staticmethod
    def set_wup(wup):
        # TODO this might not be thread safe (if the wup changes during the execution for ex)
        # ^ this should not happen in practice but we never know
        EnergyFunction.WUP = wup
        EnergyFunction.ZERO = EnergyFunction(
            [
                EnergySegment.zero(0, wup, None)
            ])
        EnergyFunction.ONE = EnergyFunction(
            [
                EnergySegment.one(0, wup, None)
            ])

    def __init__(self, segments):
        # Assuming segments are already ordered
        start = segments[0].lowerBound
        end = segments[-1].upperBound

        self.segments = segments
        self.length = len(segments)
        if start > 0:
            self.segments.insert(0, EnergySegment.zero(0, start, None))
        if end < EnergyFunction.WUP:
            self.segments.append(EnergySegment.zero(end, EnergyFunction.WUP, None))
    
    # def __init__(self, segments):
    #     self.segments = segments
    #     self.segment_map = dict([seg.lowerBound, seg] for seg in segments)

    def __str__(self):
        return " U ".join([str(seg) for seg in self.segments])

    @staticmethod
    def one():
        return EnergyFunction.ONE

    @staticmethod
    def zero():
        return EnergyFunction.ZERO

    @property
    def is_zero(self):
        # We can evaluate the segment in 0 since "normal" segments cannot get below 0
        return self.length == 1 and self.segments[0].is_zero

    ## Return the list of the points where the energy function is discontinuous.
    # In this context, "discontinuous" means that the previous energy segment and the next energy segment do not have the same equation or predecessor.
    @staticmethod
    def discontinuities(f):
        # Assuming segments are ordered
        # IMPORTANT: Global lower and upper bounds are also considered as discontinuities!
        final = [seg.lowerBound for seg in f.segments] + [f.segments[-1].upperBound]
        return final

    @property
    def domain(self):
        return (0, self.segments[-1].upperBound)

    ## Return the wup for this energy function.
    # In EnergyFunctionWup, this method is overriden to have an access to the wup that is constant in time
    def wup(self):
        return self.segments[-1].upperBound

    ## Return True if x is in the domain of this energy function (ie. [0, wup]).
    # Assuming every energy function is well-defined on [0, wup]
    def is_in_domain(self, x):
        return x >= 0 and x <= EnergyFunction.WUP

    ## Return the segment that is used when evaluating this function at x.
    def get_segment(self, x):
        assert x >= 0 and x <= EnergyFunction.WUP

        a, b = 0, self.length
        idx = int((a+b)/2)

        while True:
            seg = self.segments[idx]
            lower = seg.lowerBound
            upper = seg.upperBound
            if x == upper and upper == EnergyFunction.WUP:
                return seg
            if x < lower:
                b = idx
            elif x >= upper:
                a = idx
            else:
                return seg
            idx = int((a+b)/2)

        # for seg in self.segments:
        #     # We assume that if there is a discontinuity at x, the used segment will be the one that has maximal energy.
        #     # This means that a segment is only usable on [lowerBound, upperBound-1] unless it is the last segment (since there is no next segment to use)
        #     lower = seg.lowerBound
        #     upper = seg.upperBound
        #     if (x >= lower and x < upper) or upper == EnergyFunction.WUP:
        #         return seg

    ## Return the value of this energy function at point x.
    def evaluate(self, x):
        assert self.is_in_domain(x)
        return self.get_segment(x).evaluate(x)

    ## Merge similar energy segments.
    # TODO change the big condition in the if to use the diamond operation
    @staticmethod
    def clean(f):
        # TODO clean this!!
        new_segs = []
        # A function cannot have zero segments
        next_seg = f.segments[0]

        for i in range(1, f.length):
            old_seg = f.segments[i]

            # Merge segments with the same equation
            # This order of evaluation allows us to use the power of lazy evaluation
            # (more possible values for b)
            if old_seg.b == next_seg.b and old_seg.pred == next_seg.pred and old_seg.a == next_seg.a:
                next_seg.upperBound = old_seg.upperBound
                # print(f"merging segments, new segment: {next_seg}")
            else:
                new_segs.append(next_seg)
                next_seg = old_seg

        # Add the last remaining segment
        new_segs.append(next_seg)
        return EnergyFunction(new_segs)

    def __add__(f1, f2):
        # print(f"{f1} + {f2}")
        new_segs = []

        # The discontinuities of the max are the union of those of the 2 functions
        seen = set()
        discs = [d for d in EnergyFunction.discontinuities(f1) + EnergyFunction.discontinuities(f2) if d not in seen and not seen.add(d)]
        discs.sort()

        # Build a list of segments valid between each discontinuity
        # We use a dictionary as it is a bit more efficient
        # TODO can we put this elsewhere? We've already tried but in __init__
        segment_at_disc = {'f1': {}, 'f2': {}}
        for d in discs:
            segment_at_disc['f1'][d] = f1.get_segment(d)
            segment_at_disc['f2'][d] = f2.get_segment(d)

        for i in range(len(discs) - 1):
            lower = discs[i]
            upper = discs[i+1]

            seg1 = segment_at_disc['f1'][lower].restriction(lower, upper)
            seg2 = segment_at_disc['f2'][lower].restriction(lower, upper)

            for seg in seg1 + seg2:
                new_segs.append(seg)

            # if seg1.a == seg2.a:
            #     if seg1.b == seg2.b:
            #         next_seg = seg1 if seg2.pred is None else seg2
            #     else:
            #         next_seg = seg1 if seg1.b > seg2.b else seg2
            #     new_segs.append(next_seg.restriction(lower, upper))
            # else:
            #     # Segments might be intersecting
            #     # Use the IVT (xor)
            #     if (seg1.evaluate(lower) > seg2.evaluate(lower)) != (seg1.evaluate(upper) > seg2.evaluate(upper)):
            #         # found by manipulating the functions' equations
            #         # again, this only works if our weights are ints
            #         # intersect is guaranteed to be in ]lower, upper[ with the initial condition
            #         intersect = int((seg2.b - seg1.b) / (seg1.a - seg2.a))
            #         first_on_top = seg1 if seg1.evaluate(lower) > seg2.evaluate(lower) else seg2
            #         new_segs.append(first_on_top.restriction(lower, intersect))
            #         second_on_top = seg2 if first_on_top == seg1 else seg1
            #         new_segs.append(second_on_top.restriction(intersect, upper))
            #     else:
            #         # Segments do not intersect, we use upper because its image is guaranteed to be not equal (unless seg1 == seg2)
            #         next_seg = seg1 if seg1.evaluate(upper) > seg2.evaluate(upper) else seg2
            #         new_segs.append(next_seg.restriction(lower, upper))

        f = EnergyFunction(new_segs)
        return EnergyFunction.clean(f)

    ## Return True if this energy function is superior to the identity function.
    @property
    def is_above_one(self):
        for seg in self.segments:
            if seg.is_above_one:
                return True
        return False

    def __mul__(f1, f2):
        # TODO clean this!!
        # (even though this will have less impact on performance)
        wup = EnergyFunction.WUP
        new_segs = []

        if f1.is_zero or f2.is_zero:
            return f1.__class__.zero()

        # seen = set()
        # discs = [d for d in EnergyFunction.discontinuities(f1) + EnergyFunction.discontinuities(f2) if d not in seen and not seen.add(d)]
        # discs.sort()
        # segment_at_disc = {'f1': {}, 'f2': {}}
        # for d in discs:
        #     segment_at_disc['f1'][d] = f1.get_segment(d)
        #     segment_at_disc['f2'][d] = f2.get_segment(d)

        for seg in f1.segments:
            for next_seg in cross(seg, f2, wup):
                new_segs.append(next_seg)
            # if seg.is_zero:
            #     new_segs.append(seg)
            #     continue

            # lower = seg.lowerBound
            # upper = seg.upperBound
            # im_lower = seg.evaluate(lower)
            # im_upper = seg.evaluate(upper)

            # # Case f is constant
            # if im_lower == im_upper:
            #     next_seg = EnergySegment.const(lower,
            #                                    upper,
            #                                    f2.get_segment(im_lower).pred,
            #                                    f2.evaluate(im_lower))
            #     new_segs.append(next_seg)
            #     continue

            # # Case f is ascending
            # # For all segments of g, if the segment intersects with the image of f then apply this segment to the relevant part of the image
            # for f2_seg in f2.segments:
            #     f2_lower = f2_seg.lowerBound
            #     f2_upper = f2_seg.upperBound
            #     if f2_lower < im_upper and f2_upper > im_lower:
            #         # We need to stay in the image
            #         corr_lower = max(im_lower, f2_lower)
            #         corr_upper = min(im_upper, f2_upper)

            #         # Find the inverse by f
            #         inv_lower = int((corr_lower - seg.b) / seg.a)
            #         inv_upper = int((corr_upper - seg.b) / seg.a)

            #         new_a = seg.a * f2_seg.a
            #         new_b = f2_seg.a * seg.b + f2_seg.b
            #         next_seg = EnergySegment.incr(inv_lower, inv_upper, f2_seg.pred, new_b) if new_a == 1 else EnergySegment.const(inv_lower, inv_upper, f2_seg.pred, new_b)
            #         new_segs.append(next_seg)

        f = EnergyFunction(new_segs)
        return EnergyFunction.clean(f)

    def transition_to_sr(e, weight):
        wup = EnergyFunction.WUP
        segs = []
        if weight > 0:
            if weight >= wup:
                segs = [EnergySegment.const(0, wup, e.src, wup)]
            else:
                segs = [
                    EnergySegment.incr(0, wup - weight, e.src, weight),
                    EnergySegment.const(wup - weight, wup, e.src, wup)
                ]
        elif weight < 0:
            if weight <= -wup:
                segs = [EnergySegment.const(0, wup, e.src, -1)]
            else:
                segs = [
                    EnergySegment.const(0, -weight, e.src, -1),
                    EnergySegment.incr(-weight, wup, e.src, weight)
                ]
        else:
            segs = [
                EnergySegment.one(0, wup, e.src)
            ]
        f = EnergyFunction(segs)
        return EnergyFunction.clean(f)


## "Cross" intermediate operator
# @param seg (EnergySegment) an energy segment
# @param fun (EnergyFunction) a well-defined energy function
# @param wup (int) the wup
# @return a generator of energy segments that is the result of seg x fun
def cross(seg: EnergySegment, fun: EnergyFunction, wup: int):
    # print(f"doing {seg} x {fun}")
    # seg is the null segment on I
    if seg.a == 0 and seg.b == -1:
        yield seg
    # seg is a constant energy segment
    elif seg.a == 0 and seg.b != -1:
        yield EnergySegment.const(seg.lowerBound,
                                  seg.upperBound,
                                  fun.get_segment(seg.b).pred,
                                  fun.evaluate(seg.b))
    # seg is an increasing energy segment
    else:
        def f_inverse(x):
            return x - seg.b

        im_f = seg.image
        # the (d_i) for i in [1,n+1]
        the_ds = [seg.evaluate(seg.lowerBound)] + [disc for disc in EnergyFunction.discontinuities(fun) if disc > im_f[0] and disc < im_f[1]]
        the_ds.append(seg.evaluate(seg.upperBound))
        n = len(the_ds) - 1

        # Build the xi_i for i in [1,n]
        for i in range(n):
            left_disc = the_ds[i]
            right_disc = the_ds[i+1]
            # print(f"this is from {left_disc} to {right_disc}")
            # We don't actually need the restriction, only the equation of the underlying segment 
            r_i = fun.get_segment(left_disc)
            s = EnergySegment(f_inverse(left_disc),
                              f_inverse(right_disc),
                              r_i.pred,
                              r_i.a * seg.a,
                              r_i.b + r_i.a * seg.b)
            yield s
