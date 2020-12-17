from PyQt5 import QtGui, QtWidgets, QtCore
import math
from .utils import *


MODELS = []
PARENTS = {}
CONNECTIONS = {}
ELEMENT_TO_CONNECTION = {}


class Connection(QtWidgets.QWidget):
    def __init__(self, connect_from: QtWidgets.QWidget,
                 connect_to: QtWidgets.QWidget, parent=None):
        super(Connection, self).__init__(parent)
        self.painter = QtGui.QPainter(self)
        self.connect_from = connect_from
        self.connect_to = connect_to
        self.x1, self.y1 = None, None
        self.x2, self.y2 = None, None
        self.parent = parent
        self.active = False

        self.points = []
        self.pen = None
        self.color1 = QtGui.QColor("#e03a3c")
        self.color1.setAlpha(5)
        self.color2 = QtGui.QColor("#009ddc")
        self.color2.setAlpha(5)

        if connect_from not in CONNECTIONS.keys():
            CONNECTIONS[connect_from] = [connect_to]
        else: CONNECTIONS[connect_from].append(connect_to)
        if connect_from not in ELEMENT_TO_CONNECTION.keys():
            ELEMENT_TO_CONNECTION[connect_from] = [self]
        else: ELEMENT_TO_CONNECTION[connect_from].append(self)
        if connect_to not in ELEMENT_TO_CONNECTION.keys():
            ELEMENT_TO_CONNECTION[connect_to] = [self]
        else: ELEMENT_TO_CONNECTION[connect_to].append(self)
        if connect_to not in PARENTS.keys():
            PARENTS[connect_to] = [connect_from]
        else: PARENTS[connect_to].append(connect_from)

        self.update()
        self.show()

    def update(self) -> None:
        if self.active: self.pen = QtGui.QPen(self.color2, 20)
        else: self.pen = QtGui.QPen(self.color1, 20)
        x1, y1 = self.connect_from.center()
        x2, y2 = self.connect_to.center()
        if x1 != self.x1 and y1 != self.y1 or \
                x2 != self.x2 and y2 != self.y2:
            self.x1, self.y1 = x1, y1
            self.x2, self.y2 = x2, y2
            self.__update__(x1, y1, x2, y2)
        super(Connection, self).update()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        self.painter.begin(self)
        self.painter.setPen(self.pen)
        for i in range(0, len(self.points) - 2, 2):
            x1, y1 = self.points[i + 0:i + 2]
            x2, y2 = self.points[i + 2:i + 4]
            self.painter.drawLine(int(x1), int(y1),
                                  int(x2), int(y2))
        self.painter.end()

    def __update__(self, x1, y1, x2, y2):
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        width, height = max_x - min_x, max_y - min_y
        self.setGeometry(min_x, min_y, width, height)
        if x1 > x2: x1, x2 = 0, width
        else: x1, x2 = width, 0
        if y1 > y2: y1, y2 = 0, height
        else: y1, y2 = height, 0
        self.points = curve(x1, y1, x2, y2)

    def mouseDoubleClickEvent(self, a0: QtGui.QMouseEvent) -> None:
        x1, x2 = 0, self.width()
        y1, y2 = 0, self.height()
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        X, Y = a0.x(), k * a0.x() + b
        if Y - 50 < a0.y() < Y + 50:
            self.active = not self.active


class Connectable:
    def __init__(self):
        self.pos0 = None
        self.pos1 = None
        self.pos2 = None
        self.pos3 = None
        self.connect_to = None
        self.connect_from = None
        self.painter = QtGui.QPainter(self)
        self.min_x, self.min_y = None, None
        self.max_x, self.max_y = None, None

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        self.painter.begin(self)
        if self.pos1 and self.pos2:
            dx = self.pos1.x() - self.pos0.x()
            dy = self.pos1.y() - self.pos0.y()
            x1 = self.connect_from.x() + dx
            y1 = self.connect_from.y() + dy
            x2, y2 = self.pos2.x(), self.pos2.y()
            self.draw_curve(curve(x1, y1, x2, y2))
        if all((self.min_x, self.min_y, self.max_x, self.max_y)):
            width = self.max_x - self.min_x
            height = self.max_y - self.min_y
            self.painter.setBrush(QtGui.QColor(0, 0, 255, 50))
            self.painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255, 0)))
            self.painter.drawRect(self.min_x, self.min_y, width, height)
        self.painter.end()

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.connect_from = self.intersection(a0.pos())
        if self.connect_from is not None:
            self.pos0 = self.connect_from.pos()
        self.pos1, self.pos3 = a0.pos(), a0.pos()

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.connect_from is not None:
            self.connect_to = self.intersection(a0.pos())
            self.pos2 = a0.pos()
        selected = any((i.active for i in self.visual_elements))
        if not (self.pos2 or selected) and self.pos3:
            x1, y1 = self.pos3.x(), self.pos3.y()
            x2, y2 = a0.pos().x(), a0.pos().y()
            self.min_x, self.min_y = min(x1, x2), min(y1, y2)
            self.max_x, self.max_y = max(x1, x2), max(y1, y2)
        for element in self.visual_elements:
            element.mouseMoveEvent(a0)
        self.update()

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.connect_from and self.connect_to:
            connection = Connection(self.connect_from,
                                    self.connect_to, self)
            self.connections.append(connection)
        if all((self.min_x, self.min_y, self.max_x, self.max_y)):
            for element in self.visual_elements:
                x1, y1 = element.x(), element.y()
                x2, y2 = x1 + element.width(), y1 + element.height()
                if x1 > self.min_x and y1 > self.min_y and \
                        x2 < self.max_x and y2 < self.max_y:
                    element.active = True
                    element.update()
        self.min_x, self.min_y = None, None
        self.max_x, self.max_y = None, None
        self.pos0, self.pos1 = None, None
        self.pos2, self.pos3 = None, None
        self.connect_from = None
        self.connect_to = None
        self.update()

    def draw_curve(self, points, pen: QtGui.QPen = QtGui.QPen(QtGui.QColor(255, 0, 0, 5), 20)):
        self.painter.setPen(pen)
        for i in range(0, len(points) - 2, 2):
            x1, y1 = points[i + 0:i + 2]
            x2, y2 = points[i + 2:i + 4]
            self.painter.drawLine(int(x1), int(y1),
                                  int(x2), int(y2))

    def intersection(self, pos: QtCore.QPoint):
        if hasattr(self, 'visual_elements'):
            x, y = pos.x(), pos.y()
            for element in self.visual_elements:
                x1, y1 = element.x(), element.y()
                x2 = element.x() + element.width()
                y2 = element.y() + element.height()
                if x1 - 20 <= x <= x2 + 20 and \
                        y1 - 20 <= y <= y2 + 20:
                    return element
        return None




