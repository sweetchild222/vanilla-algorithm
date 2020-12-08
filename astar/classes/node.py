from enum import Enum

class State(Enum):
    EMPTY = 0
    OPEN  = 1
    CLOSE = 2
    BLOCK = 3


class Node:
    def __init__(self, point, h):
        self.state = State.EMPTY
        self.point = point
        self.h = h
        self.parent = None

    def getPoint(self):
        return self.point

    def setParent(self, node):
        self.parent = node

    def getParent(self):
        return self.parent

    def costH(self):
        return self.h

    def costG(self):
        return self.calcCostG(self.parent)

    def costF(self):
        return self.h + self.calcCostG(self.parent)

    def calcCostG(self, parent):

        if parent is None: #root
            return 0

        delta_x = abs(parent.point.x - self.point.x)
        delta_y = abs(parent.point.y - self.point.y)

        delta = delta_x + delta_y

        if delta == 2:
            return parent.costG() + 14
        elif delta == 1:
            return parent.costG() + 10
        else:
            print('error!!')

    def getState(self):
        if self.state == State.CLOSE:
            return 'c'
        elif self.state == State.OPEN:
            return 'o'
        elif self.state == State.BLOCK:
            return 'b'
        elif self.state == State.EMPTY:
            return 'e'

    def setClose(self):
        self.state = State.CLOSE

    def setOpen(self):
        self.state = State.OPEN

    def setBlock(self):
        self.state = State.BLOCK

    def isEmpty(self):
        return self.state == State.EMPTY

    def isBlock(self):
        return self.state == State.BLOCK

    def isOpen(self):
        return self.state == State.OPEN

    def isClose(self):
        return self.state == State.CLOSE
