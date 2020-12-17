from PyQt5 import Qt, QtGui, QtWidgets, QtCore
from .Connection import Connection
from ml import *
import random


SCALE = 1


class Interactive(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Interactive, self).__init__(parent)
        self.content = QtWidgets.QVBoxLayout(self)
        self.painter = QtGui.QPainter(self)
        self.pos1, self.pos2 = None, None

        self.connectable = True
        self.movable = True
        self.active = True

    def setGeometry(self, *args) -> None:
        if len(args) == 1: geometry = args[0]
        else: geometry = QtCore.QRect(*args)
        geometry.setWidth(self.get_size(geometry.width()))
        geometry.setHeight(self.get_size(geometry.height()))
        super(Interactive, self).setGeometry(geometry)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super(Interactive, self).paintEvent(a0)
        self.painter.begin(self)
        self.draw()
        if self.active:
            self.painter.setPen(QtGui.QPen(
                QtGui.QColor("#009ddc"), self.get_size(5)))
            self.painter.drawRect(0, 0, self.width(), self.height())
        self.painter.end()

    def center(self):
        return (self.x() + self.width() // 2,
                self.y() + self.height() // 2)

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.active and self.movable:
            self.pos1 = a0.pos()

    def mouseDoubleClickEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.active = not self.active
        self.update()

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.active and self.movable:
            if self.pos1 and self.pos2:
                dx = self.pos2.x() - self.pos1.x()
                dy = self.pos2.y() - self.pos1.y()
                self.move(self.x() + dx / 2, self.y() + dy / 2)
                self.pos1, self.pos2 = self.pos2, None
            elif self.pos1 is None: self.pos1 = a0.pos()
            elif self.pos2 is None: self.pos2 = a0.pos()
        if self.parent() is not None:
            self.parent().update()

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.pos1, self.pos2 = None, None

    @staticmethod
    def get_size(size):
        return int(size * SCALE)

    def draw(self):
        pass

    def sizes(self):
        return self.width(), self.height()
