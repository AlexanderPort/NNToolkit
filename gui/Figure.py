from PyQt5 import QtWidgets, QtGui
import math
import random
from .Movable import Movable


def random_color():
    return QtGui.QColor(
        int(random.random() * 255),
        int(random.random() * 255),
        int(random.random() * 255),
        int(random.random() * 255),
    )


class Figure(QtWidgets.QWidget, Movable):
    def __init__(self, parent=None, x=0, y=0, width=1000, height=1000, scalable=True):
        super(Figure, self).__init__(parent)
        self.setGeometry(x, y, width, height)
        self.X, self.Y, self.plots = [], [], []
        self.steps_x = 50
        self.steps_y = int(self.height() /
                           self.width() *
                           self.steps_x)
        self.painter = QtGui.QPainter(self)
        self.scalable = scalable
        self.pos0 = None
        self.pos1 = None
        self.pressed = False

    def paintEvent(self, event):
        print("paint")
        self.painter.begin(self)
        self.draw_axis(QtGui.QPen(QtGui.QColor(150, 150, 150), 3))
        self.draw_grid(QtGui.QPen(QtGui.QColor(150, 150, 150), 1))
        if any(self.plots): self.draw_plots()
        self.painter.end()

    def draw_axis(self, pen: QtGui.QPen):
        self.painter.setPen(pen)
        h_width = self.width() // 2
        h_height = self.height() // 2
        self.painter.drawLine(h_width, 0, h_width, self.height())
        self.painter.drawLine(0, h_height, self.width(), h_height)
        self.painter.drawLine(h_width, 0, h_width + 10, 40)
        self.painter.drawLine(h_width, 0, h_width - 10, 40)
        self.painter.drawLine(self.width(), h_height,
                              self.width() - 40, h_height + 10)
        self.painter.drawLine(self.width(), h_height,
                              self.width() - 40, h_height - 10)

    def draw_plots(self):
        max_x, min_x = max(self.X), min(self.X)
        max_y, min_y = max(self.Y), min(self.Y)

        if self.scalable:
            plots = []
            for i in range(len(self.plots)):
                plot = []
                print(self.plots[i])
                for j in range(0, len(self.plots[i]), 2):
                    x = self.plots[i][j + 0]
                    y = self.plots[i][j + 1]
                    if max_x - min_x != 0 and max_y - min_y != 0:
                        x = (x - min_x) / (max_x - min_x)
                        y = (y - min_y) / (max_y - min_y)
                    plot.extend([x * self.width(), y * self.height()])
                plots.append(plot)
        else:
            plots = self.plots

        for plot in plots:
            if len(plots) > 2:
                self.painter.setPen(QtGui.QPen(random_color(), 10))
                for i in range(0, len(plot) - 2, 2):
                    x1, y1 = plot[i + 0:i + 2]
                    x2, y2 = plot[i + 2:i + 4]
                    self.painter.drawLine(int(x1), int(y1),
                                          int(x2), int(y2))

    def draw_grid(self, pen: QtGui.QPen):
        self.painter.setPen(pen)
        dx = self.width() // self.steps_x
        dy = self.height() // self.steps_y
        for i in range(0, self.steps_x + 1):
            self.painter.drawLine(i * dx, 0, i * dx, self.height())
        for i in range(0, self.steps_y + 1):
            self.painter.drawLine(0, i * dy, self.width(), i * dy)

    def plot(self, plot):
        for i in range(0, len(plot), 2):
            self.X.append(plot[i + 0])
            self.Y.append(plot[i + 1])
        self.plots.append(plot)
        self.update()
