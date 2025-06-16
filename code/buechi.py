from typing import List, Dict
from dataclasses import dataclass, field
import spot

# A helper structure to preserve the additional intermediate solutions
@dataclass
class BuechiResult:
    """
    Class containing all the information needed to perform trace extraction
    after running BuechiEnergy.

    If sccEn2 and sccPred2 are None, then a trace was found without investigating
    the different states attaining an energy equal to the weak upper bound.

    If there was no viable trace, all members are None
    """
    g: spot.twa_graph = None  # Original graph
    gScc: spot.twa_graph = None  # Current accepting SCC
    renameDict: Dict[int, int] = None  # Dict from the original states to the ones in the SCC
    opts: Dict = field(default_factory=dict)  # Stores options like wup etc

    prefixEn: List[int] = None  # Optimal prefix energy for each state
    prefixPred: List[List[int]] = None  # Extended optimal predecessor list

    be: int = None  # Embedded accepting backedge

    sccEn1: List[int] = None  # Energy current SCC part 1
    sccPred1: List[List[int]] = None  # Extended optimal predecessor list current SCC part 1

    sWup: int = -1  # Maximal energy state embedded in loop

    sccEn2: List[int] = None  # Energy current SCC part 2
    sccPred2: List[List[int]] = None  # Extended optimal predecessor list current SCC part 2

    def __bool__(self) -> bool:
        return self.g is not None
