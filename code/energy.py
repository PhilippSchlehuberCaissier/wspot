from typing import List
from dataclasses import dataclass

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

    def restriction(self, low, upp):
        return EnergySegment(low, upp, self.pred, self.a, self.b)

    @staticmethod
    def nil(low, upp, pred):
        return EnergySegment(low, upp, pred, 0, -1)

    @staticmethod
    def is_nil(seg):
        return seg == EnergySegment.nil(seg.lowerBound, seg.upperBound, seg.pred)

    @staticmethod
    def identity(low, upp, pred):
        return EnergySegment(low, upp, pred, 1, 0)

    @staticmethod
    def const(low, upp, pred, k):
        return EnergySegment(low, upp, pred, 0, k)

    @staticmethod
    def incr(low, upp, pred, b):
        return EnergySegment(low, upp, pred, 1, b)


@dataclass
## Class for representing an energy function.
class EnergyFunction:
    segments: List[EnergySegment]
    # TODO maybe put the wup somewhere else, class property maybe?
    wup: int

    @staticmethod
    def identity(low, upp, wup):
        return EnergyFunction([EnergySegment.identity(low, upp, None)], wup)

    @staticmethod
    def nil(low, upp, wup):
        return EnergyFunction([EnergySegment.nil(low, upp, None)], wup)

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
        for old_seg in f.segments:
            # print(f"processing: {old_seg}")
            if next_seg is None:
                next_seg = old_seg
                continue
            # Remove zero-length segments EXCEPT if they are defined for wup
            if old_seg.lowerBound == old_seg.upperBound and old_seg.lowerBound == f.wup:
                continue

            # Merge segments with the same equation
            if old_seg.a == next_seg.a and old_seg.b == next_seg.b and old_seg.pred == next_seg.pred:
                next_seg.upperBound = old_seg.upperBound
                # print(f"merging segments, new segment: {next_seg}")
            else:
                # print(f"next segment: {next_seg}")
                new_segs.append(next_seg)
                next_seg = old_seg
        new_segs.append(next_seg)
        return EnergyFunction(new_segs, f.wup)
        # final = EnergyFunction(new_segs, f.wup)
        # return final

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

        f = EnergyFunction(new_segs, f1.wup)
        print(f"the max is {EnergyFunction.clean(f)}")
        return EnergyFunction.clean(f)

    @property
    def is_above_identity(self):
        dom = self.domain
        return EnergyFunction.max(self,
                                  EnergyFunction.identity(min(dom), max(dom), self.wup)) != EnergyFunction.identity(min(dom), max(dom), self.wup)

    def __add__(self, other):
        # TODO we don't have to fragment the other function into segments
        new_segs = []

        discs = list(set(self.discontinuities + other.discontinuities))
        discs.sort()
        print(f"new f is discontinuous at {discs}")
        for i in range(len(discs) - 1):
            lower = discs[i]
            upper = discs[i+1]

            this_seg = self.get_segment(lower)
            other_seg = other.get_segment(lower)

            # If this_seg is null then the final segment will be null since the intermediate state is not reachable
            if EnergySegment.is_nil(this_seg):
                new_segs.append(
                    EnergySegment.nil(lower,
                                      upper,
                                      None)
                    )
                continue

            # New equation is f(x) = a2(a1*x + b1) + b2
            # This is not commutative
            new_a = other_seg.a * this_seg.a
            new_b = other_seg.a * this_seg.b + other_seg.b
            print(f"f:x |--> {new_a}x + {new_b} with {this_seg} and {other_seg}")
            if new_a == 0:
                # Addition is defined if the image of the first segment is in [lower,upper]
                if min(this_seg.image) >= lower and max(this_seg.image) <= upper:
                    new_segs.append(
                        EnergySegment.const(lower,
                                            upper,
                                            other_seg.pred,
                                            min(new_b, self.wup))
                        )
                else:
                    print("Addition is not defined!")
                    new_segs.append(
                        EnergySegment.nil(lower,
                                          upper,
                                          other_seg.pred)
                        )                
            else:
                # There may be new discontinuities (result < 0 or > wup).
                # If they occur within the [lower, upper] segment then we must create other segments
                disc1 = -new_b
                disc2 = self.wup - new_b
                if disc1 > lower:
                    new_segs.append(
                        EnergySegment.const(lower,
                                            disc1,
                                            this_seg.pred,
                                            -1)
                        )
                if disc1 < disc2:
                    new_segs.append(
                        EnergySegment.incr(max(lower, disc1),
                                           min(disc2, upper),
                                           this_seg.pred,
                                           new_b)
                    )
                if disc2 < upper:
                    next_seg = EnergySegment.const(disc2,
                                                   self.wup,
                                                   this_seg.pred,
                                                   min(new_b, self.wup))
                    new_segs.append(next_seg)

        f = EnergyFunction(new_segs, self.wup)
        print(f"{self} + {other} = {f}")
        return EnergyFunction.clean(f)

    def __str__(self):
        return " U ".join([str(seg) for seg in self.segments])
