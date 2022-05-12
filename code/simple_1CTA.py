# This file holds helper functions and classes
# to parse the zone graph produced by tchecker

import parse
import math
from collections import defaultdict
from copy import deepcopy

import spot, buddy
import inspect
print(inspect.getfile(spot))

def parse_cstr(def_str):
    if "&&" in def_str:
        L1 = parse.parse("x>={}&&x<={}", def_str)
        if L1 is None:
            L1 = parse.parse("x<={}&&x>={}", def_str)
        assert L1 is not None, f"unable to parse {def_str}"
        return tuple([int(s) for s in L1])
    elif "x<=" in def_str:
        return tuple([0, int(parse.parse("x<={}", def_str)[0])])
    else:
        assert "x>=" in def_str
        return (int(parse.parse("x>={}"[0]), def_str), math.inf)

class event:
    def __init__(self, def_str:str):
        self.name = parse.parse("event:{}", def_str)[0]

class process:
    def __init__(self, def_str:str):
        self.name = parse.parse("process:{}", def_str)[0]

class location:
    def __init__(self, def_str:str):
        p0, p1 = [ss.replace("\n", "") for ss in def_str.split("{")]
        assert(p1.endswith("}"))
        p1 = p1[:-1]
        self.proc, self.loc = parse.parse("location:{}:{}", p0)

        self.initial = False
        self.invariant = (0, math.inf)
        self.rate = 0
        p1p = [ss.replace(" ", "") for ss in p1.split(":")]
        if len(p1p) >= 2:
            for i in range(0, len(p1p), 2):
                if p1p[i] == "initial":
                    self.initial = True
                elif p1p[i] == "invariant":
                    self.invariant = parse_cstr(p1p[i+1])
                elif p1p[i] == "labels":
                    lp = p1p[i+1].split(",")
                    assert len(lp) >= 1
                    self.rate = int(lp[0])
                    self.labels = lp[1:]
                else:
                    raise ValueError(f"Unknown keyword {p1p[i]}")
class edge:
    def __init__(self, def_str:str):
        p0, p1 = [ss.replace("\n", "") for ss in def_str.split("{")]
        assert(p1.endswith("}"))
        p1 = p1[:-1]
        self.proc, self.src, self.dst, self.cond = \
            parse.parse("edge:{}:{}:{}:{}", p0)

        self.reset = None
        self.guard = (0, math.inf)

        p1p = [ss.replace(" ", "") for ss in p1.split(":")]
        if len(p1p) >= 2:
            for i in range(0, len(p1p), 2):
                if p1p[i] == "do":
                    self.reset = int(parse.parse("x={}", p1p[i+1])[0])
                    assert self.reset is not None
                elif p1p[i] == "provided":
                    self.guard = parse_cstr((p1p[i+1]))
                else:
                    raise ValueError(f"Unknown keyword {p1p[i]}")

class simple_1CTA:
    # Create the class from a tchecker file
    def __init__(self, file:str):
        self.events = []
        self.processes = []
        self.locations = {}
        self.edges = {}
        self.all_propositions = set()

        with open(file, "r") as f:
            for line in f:
                line = line.strip("\n")
                if line.startswith("event"):
                    self.events.append(event(line))
                elif line.startswith("process"):
                    self.processes.append(process(line))
                elif line.startswith("location"):
                    loc = location(line)
                    self.locations.setdefault(loc.proc, {})[loc.loc] = loc
                elif line.startswith("edge"):
                    e = edge(line)
                    # "deterministic"
                    self.edges.setdefault(e.proc, {}).setdefault(e.src, {})[e.cond] = e

        all_lims = set()
        for _, proc in self.locations.items():
            for _, loc in proc.items():
                all_lims.add(loc.invariant[0])
                all_lims.add(loc.invariant[1])
                for lab in loc.labels:
                    self.all_propositions.add(lab)
        for _, proc in self.edges.items():
            for _, s in proc.items():
                for (_, e) in s.items():
                    all_lims.add(e.guard[0])
                    all_lims.add(e.guard[1])
                    self.all_propositions.add(e.cond)

        L = list(all_lims)
        L.sort()
        # We assume that all locations have invariants
        # And that the largest constant on any guard is
        # at most as large as the largest constant on some invariant
        #todo get M from tchecker? Diff to largest constant?
        #L.append(L[-1]+1)
        self.all_lims = tuple(L)
        self.all_zones = []
        self.all_delays = []
        for v, vp1 in zip(self.all_lims[:-1], self.all_lims[1:]):
            self.all_zones.append(v)
            self.all_delays.append(0) #Exact 2 left
            self.all_zones.append(v+.25) #Left open
            self.all_delays.append(vp1-v)#left 2 right
            self.all_zones.append(v+.75) #Right open
            self.all_delays.append(0)  #Right to next exact
        self.all_zones.append(self.all_lims[-1])

class zg_state:
    def __init__(self, def_str):
        p0, p1 = def_str.split("[")
        assert(p1.endswith("]"))
        p1 = p1[:-1]

        self.ostring = p1
        self.szg = int(p0)
        self.defdict = {}

        self.intval = None
        self.loc, self.zone = \
            parse.parse('intval="",vloc="{}",zone="{}"', p1)

        assert self.loc.startswith("<") and self.loc.endswith(">")
        self.loc = tuple(self.loc[1:-1].split(","))

        r1 = parse.parse("({}<=x<={})", self.zone)
        r2 = parse.parse("(x={})", self.zone)
        if r1:
            self.zone = tuple([int(s) for s in r1])  # No abstraction, zone has always 2 borders
            assert len(self.zone) == 2
        elif r2:
            L = [int(s) for s in r2]
            assert len(L) == 1
            self.zone = tuple([L[0], L[0]])

        self.locE = tuple([self.intval, self.loc])


    def __eq__(self, other):
        return self.defdict == other.defdict



class ZG2HOABuilder:
    def __init__(self, sCTA:simple_1CTA, zg_file:str):
        self.sCTA = sCTA
        self.wTWA = spot.make_twa_graph(spot.make_bdd_dict())

        # Register all the proposition
        self.propDict = dict()
        self.falseCond = buddy.bddtrue
        for prop in self.sCTA.all_propositions:
            p = buddy.bdd_ithvar(self.wTWA.register_ap(prop))
            self.propDict[prop] = p
            self.falseCond = buddy.bdd_and(self.falseCond, buddy.bdd_not(p))
        assert "dT" not in self.propDict.keys()
        p = buddy.bdd_ithvar(self.wTWA.register_ap("dT"))
        self.propDict["dT"] = p
        self.dTcond = buddy.bdd_and(self.falseCond, p)
        self.falseCond = buddy.bdd_and(self.falseCond, buddy.bdd_not(p))

        self.weigths = [0] #Zero edge weight

        self.szgdict = dict()
        self.locE2scpa = dict()
        self.szg2scpa = dict()
        self.scpa2stwa = dict()
        self.scp2stwa = dict()
        self.locE2info = dict()

        self.all_edges = set()

        with open(zg_file, "r") as f:
            for line in f:
                line = line.strip("\n")
                line = line.replace(" ", "")

                if (line[0] < "0") or (line[0] > "9"):
                    continue

                is_edge = "->" in line.split("[")[0]
                if is_edge:
                    self.add_edge(line)
                else:
                    self.add_state(line)
        for idx, w in enumerate(self.weigths):
            # todo
            if idx != 0:
                spot.set_weight(self.wTWA, idx, w)
        # Set the initial state
        init = None
        for ns, zgstate in self.szgdict.items():
            possible_init = True
            for idx, l in enumerate(zgstate.loc):
                if not self.sCTA.locations[self.sCTA.processes[idx].name][l].initial:
                    possible_init = False
                    break
            if not possible_init:
                continue
            assert init is None, "HOA does not support multiple initial states"
            if self.szg2scpa[zgstate.szg][0] != -1:
                init = self.szg2scpa[zgstate.szg][0]
        self.wTWA.set_init_state(init)

    def add_edge(self, def_str):
        p0, p1 = def_str.split("[")
        assert p1.endswith("]")
        p1 = p1[:-1]

        src, dst = [int(s) for s in parse.parse("{}->{}", p0)]
        zsrc = self.szgdict[src]
        zdst = self.szgdict[dst]
        assert zsrc.szg == src
        assert zdst.szg == dst

        all_edges = [tuple(s.split("@")) for s in parse.parse('vedge="<{}>"', p1)[0].split(",")]

        all_edge_prop = deepcopy(zsrc.all_labels)

        guard = [0, math.inf]
        reset = []

        for procname, eve in all_edges:
            procidx = [p.name for p in self.sCTA.processes].index(procname)
            assert procidx != -1
            e = self.sCTA.edges[procname][zsrc.loc[procidx]][eve]
            if e.reset is not None:
                reset.append(e.reset)
            guard = [max(guard[0], e.guard[0]), min(guard[1], e.guard[1])]
            all_edge_prop.add(e.cond)
        assert guard[0] <= guard[1]
        assert len(reset) <= 1 or len(set(reset)) == 1
        if reset:
            reset = reset[0]

        # The actual condition is the conjunction of all prop that appear and
        # and the negation of those who do not appear
        econd = buddy.bddtrue
        for prop, var in self.propDict.items():
            if prop in all_edge_prop:
                econd = buddy.bdd_and(econd, var)
            else:
                econd = buddy.bdd_and(econd, buddy.bdd_not(var))

        dstinv = self.locE2info[zdst.locE]["inv"]
        if isinstance(reset, int):
            # If there is a reset, it needs to be compatible with invariant of dst
            assert dstinv[0] <= reset <= dstinv[1]
            dstCPAidx = self.sCTA.all_zones.index(reset)
            assert dstCPAidx != -1

        # corner abstraction valid?
        # (compatible with guards)
        lsrczSCPA = self.szg2scpa[src]
        ldstzSCPA = self.szg2scpa[dst]
        for i, zv in enumerate(self.sCTA.all_zones):
            if (lsrczSCPA[i] != -1) and guard[0] <= zv <= guard[1]:
                # Abstraction compatible with guard
                if isinstance(reset, int):
                    # todo cond
                    if (ldstzSCPA[dstCPAidx] != -1) and not (lsrczSCPA[i], ldstzSCPA[dstCPAidx]) in self.all_edges:
                        self.all_edges.add((lsrczSCPA[i], ldstzSCPA[dstCPAidx]))
                        self.wTWA.new_edge(lsrczSCPA[i], ldstzSCPA[dstCPAidx], econd)
                        self.weigths.append(0)
                else:
                    # Action transition, stay in the same interval, if allowed
                    if ldstzSCPA[i] != -1:
                        if not (lsrczSCPA[i], ldstzSCPA[i]) in self.all_edges:
                            self.all_edges.add((lsrczSCPA[i], ldstzSCPA[i]))
                            self.wTWA.new_edge(lsrczSCPA[i], ldstzSCPA[i], econd)
                            self.weigths.append(0)

    def add_state(self, def_str:str):
        zgstate = zg_state(def_str)
        assert zgstate.szg not in self.szgdict.keys()
        assert zgstate not in self.szgdict.items()

        # Explode the state (without zone)
        # into the corner abstractions
        if not zgstate.locE in self.locE2scpa.keys():
            # Conjunction of invariants -> interval, non-empty or not in ZG
            # Get the combined rate #todo
            inv = [0, math.inf]
            all_rate = []
            rate = 0
            zgstate.all_labels = set()
            for i,loc in enumerate(zgstate.loc):
                ll = self.sCTA.locations[self.sCTA.processes[i].name][loc]
                rate = rate + ll.rate
                all_rate.append(ll.rate)
                inv = [max(inv[0], ll.invariant[0]), min(inv[1], ll.invariant[1])]
                zgstate.all_labels = zgstate.all_labels.union(ll.labels)
            assert inv[0] <= inv[1]
            self.locE2info[zgstate.locE] = {"inv":tuple(inv), "rate":tuple(all_rate)}

            lSCPA = self.locE2scpa[zgstate.locE] = len(self.sCTA.all_zones)*[-1]
            for i,zv in enumerate(self.sCTA.all_zones):
                if not (inv[0] <= zv and zv <= inv[1]):
                    continue
                lSCPA[i] = self.wTWA.new_state()
            # Create all the transitions
            # todo cond <-> labels
            for i in range(len(self.sCTA.all_zones)-1):
                src = lSCPA[i]
                dst = lSCPA[i+1]
                zi = self.sCTA.all_zones[i]
                zip1 = self.sCTA.all_zones[i+1]
                is_dT = (0.15 < zi%1 and zi%1 < 0.35) and (0.65 < zip1%1 and zip1%1 < 0.85)

                if src == -1 or dst == -1:
                    continue
                self.wTWA.new_edge(src, dst, self.dTcond if is_dT else self.falseCond) #todo need color here?
                self.weigths.append(rate*self.sCTA.all_delays[i])
        else:
            # The labels of this location have already been explored
            # we can compy them, as only the zone changes
            # All abstractions have been generates
            found = False
            for szg in self.szgdict.values():
                if szg.loc == zgstate.loc:
                    found = True
                    zgstate.all_labels = deepcopy(szg.all_labels)
                    break
            assert found

        # Add it to the dict
        # Note do this afterwards, to avoid copying non-existing attribute
        self.szgdict[zgstate.szg] = zgstate

        lSCPA = self.locE2scpa[zgstate.locE]
        lzSCPA = self.szg2scpa[zgstate.szg] = len(lSCPA)*[-1]
        for i, zv in enumerate(self.sCTA.all_zones):
            if not (zgstate.zone[0] <= zv and zv <= zgstate.zone[1]):
                continue
            assert lSCPA[i] != -1
            lzSCPA[i] = lSCPA[i]





















