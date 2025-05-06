## @package ipython_utils
# Utility functions for printing various information in WBA_utils.

from typing import List


## Class that implements the Singleton design pattern.
class IPythonUtilsMeta(type):
    _instances = {}

    def __call__(c, *args, **kwargs):
        if c not in c._instances:
            new_instance = super().__call__(*args, **kwargs)
            c._instances[c] = new_instance
        return c._instances[c]


## Singleton class to be called when printing information.
class IPythonUtils(metaclass=IPythonUtilsMeta):
    do_display = 0

    def set_display_mode(self, do_display):
        self.do_display = do_display
        return

    def print_c(self, *args, **kwargs):
        if self.do_display > 0:
            print(*args, **kwargs)
        return

    def display_c(self, aut, opt=""):
        if self.do_display > 1:
            display(aut.show(opt))
        return

    def highlight_c(self,
                    aut: "spot.twa_graph",
                    pred: "List[List[int]]",
                    predColors: List[int] = [1, 2, 3, 4, 5],
                    opt="") -> None:
        """
        Args:
        aut: The automaton
        pred: List of lists. The list pred[s] contains all predecessors for state s
        predColors: How many predecessors should be colored and how. The -ith predecessor will be colored with the i-1th color
        opt: Additional options passed to highlight_edges

        Returns: None

        """
        if self.do_display > 1:
            # Create a list of edges for each color
            cDict = dict(zip(predColors, [[] for _ in predColors]))
            for s in range(aut.num_states()):
                for c, en in zip(predColors, reversed(pred[s])):
                    cDict[c].append(en)
            for c, edges in cDict.items():
                aut.highlight_edges(edges, c)
        self.display_c(aut, opt)
