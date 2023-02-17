import numpy as np


class Layer:

    def __init__(self, width) -> None:
        self.width = width
        self.nodes = np.zeros(width)
    



