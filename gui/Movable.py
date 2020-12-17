from PyQt5 import QtGui, QtCore


class Movable:
    def __init__(self):
        self.movable = False
        self.move0 = None
        self.move1 = None

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.movable = not self.movable
        self.update()

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.movable = False
        self.move1 = None
        self.move0 = None
        self.update()

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.movable:
            if self.move0 and self.move1:
                dx = self.move1.x() - self.move0.x()
                dy = self.move1.y() - self.move0.y()
                self.move(self.x() + dx / 2, self.y() + dy / 2)
                self.move0 = self.move1
                self.move1 = None
            elif self.move0 is None:
                self.move0 = a0.pos()
            elif self.move1 is None:
                self.move1 = a0.pos()

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        print(343434)
        if a0.key() == QtCore.Qt.Key_Delete:
            if self.movable: self.deleteLater()
