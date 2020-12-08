from PIL import Image, ImageDraw, ImageFont
from classes.point import Point
import os

step = 0
RectSize = 120

def draw(map, start, stop):

    global RectSize

    width = map.width
    height = map.height

    image = Image.new('RGB', (width * RectSize, height * RectSize), color = 'white')
    drawer = ImageDraw.Draw(image)

    for y in range(height):
        for x in range(width):
            node = map.getNode(Point(x, y))
            point = Point(x * RectSize, (height - y - 1) * RectSize)

            drawRect(node, point, drawer)

            if node.getPoint() == start:
                drawText(node, point, drawer, 'S')
            elif node.getPoint() == stop:
                drawText(node, point, drawer, 'E')

            if node.getState() == 'c' or node.getState() == 'o':
                drawCost(node, point, drawer)
                drawTriangle(node, point, drawer)

    saveFile(image)


def saveFile(image):

    global step

    folder = 'result'

    if os.path.isdir(folder) is False:
        os.mkdir(folder)

    image.save(folder + '/' + str(step) + '.png')

    step += 1


def drawCost(node, point, drawer):

    global RectSize

    padding = RectSize * 0.05

    font = ImageFont.truetype("arial.ttf", 18)
    cost = str(node.costG()) + ' + ' + str(node.costH()) + ' = ' + str(node.costG() + node.costH())

    drawer.text((point.x + padding, point.y + padding), cost, font=font, fill=(0, 0, 0))


def drawText(node, point, drawer, text):

    global RectSize

    fontSize = 18

    font = ImageFont.truetype("arial.ttf", 18)

    drawer.text((point.x + 5, point.y +  RectSize - (fontSize) - 5), text, font=font, fill=(255, 0, 255))

def drawRect(node, point, drawer):

    global RectSize

    colorMap = {'c':'green', 'b':'black', 'o':'orange', 'e':'white'}

    drawer.rectangle([(point.x, point.y), (point.x + RectSize, point.y + RectSize)], colorMap[node.getState()], 'black',  1)


def drawRectColor(node, point, drawer, color):

    global RectSize

    drawer.rectangle([(point.x, point.y), (point.x + RectSize, point.y + RectSize)], color, 'black',  1)


def drawTriangle(node, point, drawer):

    if node.getParent() is None:
        return

    global RectSize

    centerX = point.x + (RectSize / 2)
    centerY =  point.y + (RectSize / 2) + 15

    crossLength = RectSize * 0.25

    angleLength = crossLength * 0.707

    lineStart = None
    lineEnd = None

    diff = node.getPoint() - node.getParent().getPoint()

    diffList = {str(Point(1, 0)):[Point(-crossLength, 0), Point(crossLength, 0)],
                str(Point(-1, 0)):[Point(crossLength, 0), Point(-crossLength, 0)],
                str(Point(0, 1)):[Point(0, crossLength), Point(0, -crossLength)],
                str(Point(0, -1)):[Point(0, -crossLength), Point(0, crossLength)],
                str(Point(1, 1)):[Point(-angleLength, angleLength), Point(angleLength, -angleLength)],
                str(Point(1, -1)):[Point(-angleLength, -angleLength), Point(angleLength, angleLength)],
                str(Point(-1, -1)):[Point(angleLength, -angleLength), Point(-angleLength, angleLength)],
                str(Point(-1, 1)):[Point(angleLength, angleLength), Point(-angleLength, -angleLength)]}

    findDiff = diffList[str(diff)]

    if findDiff is None:
        return

    lineStart = findDiff[0] + Point(centerX, centerY)
    lineEnd = findDiff[1] + Point(centerX, centerY)

    if lineStart is not None and lineEnd is not None:
        radius = 5
        leftUp = (lineStart.x - radius, lineStart.y - radius)
        rightDown = (lineStart.x + radius, lineStart.y + radius)

        drawer.line([(lineStart.x, lineStart.y), (lineEnd.x, lineEnd.y)], fill ="red", width = 4)
        drawer.ellipse([leftUp, rightDown], fill = 'red')
