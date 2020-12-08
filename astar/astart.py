
from classes.node import Node
from classes.openList import OpenList
from classes.point import Point
from classes.map import Map
import classes.drawer as drawer

def aStar(start, stop, nodeMap):

    openList = OpenList()

    node = map.getNode(start)

    while True:

        node.setClose()

        drawer.draw(map, start, stop)

        if node.getPoint() == stop:
            return node

        childList = lookAround(node, nodeMap)

        openList.append(childList)

        node = openList.minCostFNode()

        if node is None:
            return None

def neighborBlock(node, deltaPoint, map):

    x = deltaPoint.x
    y = deltaPoint.y

    neighborList = []

    if (abs(x) + abs(y)) == 2:
        neighborList = [Point(x, 0), Point(0, y)]

    for point in neighborList:

        neighborPoint = node.getPoint() + point
        neighbor = map.getNode(node.getPoint() + point)

        if neighbor is None:
            continue

        if neighbor.isBlock() is True:
            return True

    return False


def lookAround(node, map):

    childDelta = [Point(1, 0), Point(1, -1), Point(0, -1), Point(-1, -1),
                 Point(-1, 0), Point(-1, 1), Point(0, 1), Point(1, 1)]

    openList = []

    for delta in childDelta:

        childPoint = node.getPoint() + delta
        childNode = map.getNode(childPoint)

        if childNode is None:
            continue

        if neighborBlock(node, delta, map) is True:
            continue

        if childNode.isBlock():
            continue
        elif childNode.isClose():
            continue
        elif childNode.isEmpty():
            childNode.setParent(node)
            childNode.setOpen()
            openList.append(childNode)
        elif childNode.isOpen():
            currentCostG = childNode.costG()
            newCostG = childNode.calcCostG(node)
            if currentCostG > newCostG:
                childNode.setParent(node)
        else:
            print('error!')

    return openList


"""
data = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0]
"""

"""
data = [
    0, -1, -1, -1, 0, 0, 0, 0,
    -1, 0, 0, 0, -1, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0,
    0, -1, 0, 0, -1, 0, 0, 0,
    0, 0, 0, -1, 0, 0, 0, 0]
"""


data = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0]


width = 8
height = 6

start = Point(2, 3)
stop = Point(6, 3)

map = Map(data, width, height, stop)
node = aStar(start, stop, map)

while node:
    print(node.getPoint())
    node = node.getParent()
