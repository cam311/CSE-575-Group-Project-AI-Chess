import numpy as np

class Encoder():
    def __init__(self):
        codes, i = {}, 0

        for nSquares in range(1,8):
            for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
                codes[(nSquares,direction)] = i
                i += 1

        for two in ["N","S"]:
            for one in ["E","W"]:
                codes[("knight", two, one)] , i = i , i + 1
        for two in ["E","W"]:
            for one in ["N","S"]:
                codes[("knight", two, one)] , i = i , i + 1

        for move in ["N","NW","NE"]:
            for promote_to in ["Rook","Knight","Bishop"]:
                codes[("underpromotion", move, promote_to)] , i = i , i + 1

        self.codes = codes
