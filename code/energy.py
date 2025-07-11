from typing import List
from dataclasses import dataclass

from semiring import Semiring


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
        return (self.lowerBound, self.upperBound)

    @property
    def image(self):
        return (self.b, self.b) if self.a == 0 else (self.lowerBound + self.b, self.upperBound + self.b)

    def __str__(self):
        str_def = f"[{self.lowerBound}, {self.upperBound}] -> IR"
        str_a = 'e_in' if self.a else ''
        str_b = '' if self.a and not self.b else str(self.b) if not self.a else f" + {self.b}"
        return f"{str_def} ; e_in |---> {str_a}{str_b}" + f" (from {self.pred})"

    def is_in_domain(self, x):
        return x >= self.domain[0] and x <= self.domain[1]

    def evaluate(self, e_in):
        assert self.is_in_domain(e_in)
        return self.a * e_in + self.b

    @property
    def is_above_one(self):
        return self.evaluate(self.lowerBound) >= self.lowerBound and self.pred is not None

    def restriction(self, low, upp):
        # TODO do we really need to restrict energy segments?
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


@dataclass
## Class for representing an energy function.
class EnergyFunction(Semiring):
    segments: List[EnergySegment]

    def __str__(self):
        return " U ".join([str(seg) for seg in self.segments])

    @staticmethod
    def one(wup):
        return EnergyFunction([EnergySegment.one(0, wup, None)])

    @staticmethod
    def zero(wup):
        return EnergyFunction([EnergySegment.zero(0, wup, None)])

    @property
    def is_zero(self):
        # We can evaluate the segment in 0 since "normal" segments cannot get below 0
        return len(self.segments) == 1 and self.segments[0].evaluate(0) == -1

    @property
    def discontinuities(self):
        # Assuming segments are ordered
        # IMPORTANT: Global lower and upper bounds are also considered as discontinuities!
        final = [seg.lowerBound for seg in self.segments] + [self.segments[-1].upperBound]
        return final

    @property
    def domain(self):
        return (0, self.segments[-1].upperBound)

    @property
    def wup(self):
        return self.domain[1]

    def is_in_domain(self, x):
        return x >= self.domain[0] and x <= self.domain[1]

    ## Return the segment that is used when evaluating this function at x.
    def get_segment(self, x):
        assert self.is_in_domain(x)

        # TODO dichotomy?
        for seg in self.segments:
            # We assume that if there is a discontinuity at x, the used segment will be the one that has maximal energy.
            # This means that a segment is only usable on [lowerBound, upperBound-1] unless it is the last segment (since there is no next segment to use)
            if (x >= seg.lowerBound and x < seg.upperBound) or seg == self.segments[-1]:
                return seg

    def evaluate(self, x):
        assert self.is_in_domain(x)
        return self.get_segment(x).evaluate(x)

    @staticmethod
    def clean(f):
        # TODO clean this!!
        wup = f.domain[1]

        new_segs = []
        next_seg = None
        # Whether we need to make another pass
        to_clean = False

        # f must be defined on [0, wup]
        # if f.segments[0].lowerBound > 0:
        #     f.segments.insert(0, EnergySegment.zero(0,
        #                                             f.segments[0].lowerBound,
        #                                             None))
        # if f.segments[-1].upperBound < wup:
        #     f.segments.append(EnergySegment.zero(f.segments[-1].upperBound,
        #                                          wup,
        #                                          None))

        for old_seg in f.segments:
            # Keep every segment defined on an interval of [0, wup]
            old_seg = old_seg.restriction(0, wup)

            # # Remove duplicate segments
            # if old_seg == next_seg or old_seg in new_segs:
            #     next_seg = None
            #     continue

            # # Remove zero-length segments EXCEPT if they are defined for wup
            # if old_seg.lowerBound == old_seg.upperBound and old_seg.lowerBound != WUP.get_wup():
            #     continue

            if next_seg is None:# or old_seg.lowerBound > old_seg.upperBound:
                next_seg = old_seg
                continue

            # # Cap every segment to wup
            # if old_seg.evaluate(old_seg.upperBound) > wup:
            #     if next_seg is not None:
            #         new_segs.append(next_seg)
            #         next_seg = None
            #     to_clean = True

            #     if old_seg.a == 0:
            #         new_segs.append(EnergySegment.const(old_seg.lowerBound,
            #                                             old_seg.upperBound,
            #                                             old_seg.pred,
            #                                             wup
            #                                             ))
            #     else:
            #         disc = int((wup - old_seg.b) / old_seg.a)
            #         new_segs.append(EnergySegment.incr(old_seg.lowerBound,
            #                                            disc,
            #                                            old_seg.pred,
            #                                            old_seg.b
            #                                            ))
            #         new_segs.append(EnergySegment.const(disc,
            #                                             old_seg.upperBound,
            #                                             old_seg.pred,
            #                                             wup
            #                                             ))
            #     continue

            # # Same procedure: nullify non-null segments below zero
            # if old_seg.evaluate(old_seg.lowerBound) < 0 and not old_seg.is_zero:
            #     if next_seg is not None:
            #         new_segs.append(next_seg)
            #         next_seg = None
            #     to_clean = True

            #     if old_seg.a == 0:
            #         new_segs.append(EnergySegment.zero(old_seg.lowerBound,
            #                                            old_seg.upperBound,
            #                                            None
            #                                            ))
            #     else:
            #         disc = int(-old_seg.b / old_seg.a)
            #         new_segs.append(EnergySegment.zero(old_seg.lowerBound,
            #                                            disc,
            #                                            None
            #                                            ))
            #         new_segs.append(EnergySegment.incr(disc,
            #                                            old_seg.upperBound,
            #                                            old_seg.pred,
            #                                            old_seg.b
            #                                            ))
            #     continue

            # Merge segments with the same equation
            if old_seg.a == next_seg.a and old_seg.b == next_seg.b and old_seg.pred == next_seg.pred:
                next_seg.upperBound = old_seg.upperBound
                # print(f"merging segments, new segment: {next_seg}")
            else:
                new_segs.append(next_seg)
                next_seg = old_seg
        if next_seg is not None:
            new_segs.append(next_seg)
        f = EnergyFunction(new_segs)
        # print(f"final is {f}")
        return EnergyFunction.clean(f) if to_clean else f

    def __add__(f1, f2):
        # print(f"{f1} + {f2}")
        new_segs = []

        # The discontinuities of the max are the union of those of the 2 functions
        seen = set()
        discs = [d for d in f1.discontinuities + f2.discontinuities if d not in seen and not seen.add(d)]
        discs.sort()

        # Build a list of segments valid at each discontinuity
        # We use a dictionary as it is a bit more efficient
        segment_at_disc = {'f1': {}, 'f2': {}}
        for d in discs:
            segment_at_disc['f1'][d] = f1.get_segment(d)
            segment_at_disc['f2'][d] = f2.get_segment(d)

        for i in range(len(discs) - 1):
            lower = discs[i]
            upper = discs[i+1]

            seg1 = segment_at_disc['f1'][lower]
            seg2 = segment_at_disc['f2'][lower]

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
        return EnergyFunction.clean(f)

    @property
    def is_above_one(self):
        for seg in self.segments:
            if seg.is_above_one:
                return True
        return False

    def __mul__(self, other):
        # TODO clean this!!
        wup = self.domain[1]
        new_segs = []

        if self.is_zero or other.is_zero:
            return EnergyFunction.zero(wup)

        for seg in self.segments:
            if seg.is_zero:
                new_segs.append(seg)
                continue

            lower = seg.lowerBound
            upper = seg.upperBound
            im_lower = seg.evaluate(lower)
            im_upper = seg.evaluate(upper)

            # Case f is constant
            if im_lower == im_upper:
                next_seg = EnergySegment.const(lower,
                                               upper,
                                               other.get_segment(im_lower).pred,
                                               other.evaluate(im_lower))
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
                    next_seg = EnergySegment.incr(inv_lower, inv_upper, other_seg.pred, new_b) if new_a == 1 else EnergySegment.const(inv_lower, inv_upper, other_seg.pred, new_b)
                    new_segs.append(next_seg)

        f = EnergyFunction(new_segs)
        return EnergyFunction.clean(f)

    def transition_to_sr(wup, e, weight):
        segs = []
        if weight > 0:
            segs = [
                EnergySegment.incr(0, wup - weight, e.src, weight),
                EnergySegment.const(wup - weight, wup, e.src, wup)
            ]
        elif weight < 0:
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


## Class for energy functions parametrized with a wup
def EnergyFunctionWup(wup):
    return type(
        "EnergyFunctionWup",
        (EnergyFunction, ),
        {"wup": wup,
         "zero": lambda: EnergyFunction.zero(wup),
         "one": lambda: EnergyFunction.one(wup),
         "transition_to_sr": lambda e, weight: EnergyFunction.transition_to_sr(wup, e, weight)}
    )
