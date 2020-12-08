class OpenList:

    def __init__(self):
        self.list = []

    def append(self, list):
        self.list = self.list + list

    def minCostFNode(self):

        minNode = None
        minIndex = -1
        index = 0

        for node in self.list:

            if minNode == None:
                minIndex = index
                minNode = node

            if minNode.costF() >= node.costF():
                minIndex = index
                minNode = node

            index += 1

        if minIndex != -1:
            del self.list[minIndex]

        return minNode
