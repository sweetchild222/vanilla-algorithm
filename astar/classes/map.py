from classes.node import Node
from classes.point import Point

class Map:
    def __init__(self, data, width, height, stop):

        self.map = []
        self.width = width
        self.height = height

        index = 0

        for d in data:
            x = index % self.width
            y = height - 1 - (index // self.width)
            point = Point(x, y)
            h = (abs(stop.x - point.x) + abs(stop.y - point.y)) * 10

            node = Node(point, h)

            if d == -1:
                node.setBlock()

            self.map.append(node)

            index += 1

    def getNode(self, point):
        if point.x < 0 or point.y < 0:
            print('over', point.y, ', ', point.x)
            return None

        if point.x < self.width and point.y < self.height:
            index = point.x + ((self.width * self.height) - ((point.y + 1) * self.width))
            return self.map[index]
        else:
            print('over', point.y, ', ', point.x)
            return None

    def show(self):

        text = []

        for y in range(self.height):
            line = []
            for x in range(self.width):
                node = self.getNode(Point(x, self.height - y - 1))

                if node.getState() == 'o':
                    line.append(str(node.costG()) + ",  ")
                else:
                    line.append(str(node.getState()) + ",  ")

            line.append('\n')

            text += line

        print("".join(text))
