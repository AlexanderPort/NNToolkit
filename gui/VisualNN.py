from PyQt5 import QtWidgets, QtGui, QtCore
from ml import *
import math
from PyQt5.Qt import Qt
from .Figure import *
from .Image import Image


class VisualNN(QtWidgets.QWidget):
    def __init__(self, model: Model, parent=None):
        super(VisualNN, self).__init__(parent)
        self.setGeometry(100, 100, 10000, 10000)
        self.model = model
        self.figure = Figure(self, 1500, 100, 1000, 1000)
        self.input_data, self.output_data = None, None
        self.load_input_data = QtWidgets.QPushButton("load input data", self)
        self.load_input_data.clicked.connect(self.get_input_data)
        self.load_output_data = QtWidgets.QPushButton("load output data", self)
        self.load_output_data.clicked.connect(self.get_output_data)
        self.load_input_data.move(300, 0)
        self.run = QtWidgets.QPushButton("Optimize", self)
        self.run.clicked.connect(self.update_model)
        self.run.move(600, 0)

        self.optimizer = Optimizer(model.parameters, learning_rate=10)
        self.loss = MSELoss()
        self.loss_data = []

        self.X, self.Y, self.R = 50, 100, 40
        self.dx, self.dy = 200, 50
        self.painter = QtGui.QPainter(self)
        self.min_weight = self.get_min_weight()
        self.max_weight = self.get_max_weight()
        self.pos0, self.pos1 = None, None
        self.iteration = 1
        self.image = None

    def draw_weights(self, layer: Dense):
        coords1, coords2 = [], []
        indexes1, indexes2 = [], []
        num_inputs = layer.weights.shape[0]
        num_outputs = layer.weights.shape[1]
        gray_brush = QtGui.QBrush(QtGui.QColor("gray"))
        black_brush = QtGui.QBrush(QtGui.QColor("black"))
        for i in range(num_inputs % 21):
            indexes1.append(i)
            coords1.append([self.X, self.Y + self.dy * i])
            self.draw_circle(self.X, self.Y + self.dy * i,
                             self.R, brush=gray_brush)
        if num_inputs - 20 > 0:
            for i in range(3):
                self.draw_circle(self.X, self.Y + self.dy * (i + 18),
                                 5, brush=black_brush)
            indexes1.append(num_inputs - 1)
            coords1.append([self.X, self.Y + self.dy * 23])
            self.draw_circle(self.X, self.Y + self.dy * 23,
                             self.R, brush=gray_brush)
        self.X += self.dx
        for i in range(num_outputs % 21):
            indexes2.append(i)
            coords2.append([self.X + self.dx, self.Y + self.dy * i])
            self.draw_circle(self.X + self.dx, self.Y + self.dy * i,
                             self.R, brush=QtGui.QBrush(QtGui.QColor("gray")))

        if num_outputs - 20 > 0:
            for i in range(3):
                self.draw_circle(self.X + self.dx, self.Y + self.dy * (i + 18),
                                 5, brush=QtGui.QBrush(QtGui.QColor("black")))
            indexes2.append(num_inputs - 1)
            coords2.append([self.X + self.dx, self.Y + self.dy * 23])
            self.draw_circle(self.X + self.dx, self.Y + self.dy * 23,
                             self.R, brush=QtGui.QBrush(QtGui.QColor("gray")))

        weights = layer.weights.data
        for i in range(len(coords1)):
            for j in range(len(coords2)):
                self.draw_weight(*coords1[i], *coords2[j],
                                 float(weights[indexes1[i]][indexes2[j]]))

    def paintEvent(self, event):
        self.painter.begin(self)
        x = self.X
        for i in range(0, len(self.model.layers)):
            if type(self.model.layers[i]) == Dense:
                self.draw_weights(self.model.layers[i])
                self.X += self.dx
        self.X = x
        self.painter.end()

    def draw_weight(self, x1, y1, x2, y2, weight):
        d = self.normalize(weight)
        pen = QtGui.QPen(QtGui.QColor(255 * d, 255 * d,
                                      255 * d, 255 * d), 1)

        if weight >= 0:
            pen = QtGui.QPen(QtGui.QColor(255 * d, 0,
                                          0, 255 * d / 3), 3 * d)
        elif weight < 0:
            pen = QtGui.QPen(QtGui.QColor(0, 0,
                                          255 * d, 255 * d / 3), 3 * d)
        self.painter.setPen(pen)
        self.draw_line(x1, y1, x2, y2, pen)
        # self.draw_curve(self.get_curve(x1, y1, x2, y2), pen)

    def normalize(self, weight):
        return (weight - self.min_weight) / \
               (self.max_weight - self.min_weight)

    def draw_curve(self, plot, pen: QtGui.QPen=QtGui.QPen(QtGui.QColor())):
        self.painter.setPen(pen)
        for i in range(0, len(plot) - 2, 2):
            x1, y1 = plot[i + 0:i + 2]
            x2, y2 = plot[i + 2:i + 4]
            self.painter.drawLine(int(x1), int(y1),
                                  int(x2), int(y2))

    def draw_line(self, x1, y1, x2, y2, pen: QtGui.QPen):
        self.painter.setPen(pen)
        self.painter.drawLine(x1, y1, x2, y2)

    def draw_circle(self, x, y, r, pen: QtGui.QPen = None, brush: QtGui.QBrush = None):
        if pen: self.painter.setPen(pen)
        if brush: self.painter.setBrush(brush)
        self.painter.drawEllipse(x - r / 2, y - r / 2, r, r)

    @staticmethod
    def get_curve(x1, y1, x2, y2):
        mx = (x2 - x1) / 2
        my = (y2 - y1) / 2
        c1 = [x1, y1 + my]
        c2 = [x2, y2 - my]
        steps = 0.01
        plot = []

        angle = 1.5 * math.pi
        goal = 2 * math.pi
        while angle < goal:
            x = c1[0] + mx * math.cos(angle)
            y = c1[1] + my * math.sin(angle)
            plot.extend([x, y])
            angle += steps

        angle = 1 * math.pi
        goal = 0.5 * math.pi
        while angle > goal:
            x = c2[0] + mx * math.cos(angle)
            y = c2[1] + my * math.sin(angle)
            plot.extend([x, y])
            angle -= steps

        return plot

    def get_min_weight(self):
        return min([weights.min() for weights in self.model.parameters])

    def get_max_weight(self):
        return max([weights.max() for weights in self.model.parameters])

    def directDraw(self, rect):
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0, QtGui.QColor("#0D324D"))
        gradient.setColorAt(0.5, QtGui.QColor("#7F5A83"))
        self.painter.fillRect(rect, gradient)

    def get_input_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self)

        try:
            if "jpg" in path or "png" in path or "jpeg" in path:
                self.image = Image(path, self)
                self.image.move(0, 1000)
            self.input_data = Tensor.load(path)
            print("input")
        except:
            pass

    def get_output_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self)
        try:
            self.output_data = Tensor.load(path)
            print("output")
        except:
            pass

    def update_model(self):
        try:
            pred = self.model(self.input_data)
            loss = self.loss(self.output_data, pred)
            loss.backward()
            mean_loss = loss.mean()
            print(mean_loss)
            self.loss_data.append(self.iteration)
            self.loss_data.append(mean_loss)
            self.iteration += 1
            self.figure.plot(self.loss_data)
            self.optimizer.step()
            self.update()
        except:
            pass


class ScrollableVisualNN(QtWidgets.QWidget):
    def __init__(self, model: Model):
        super(ScrollableVisualNN, self).__init__()
        self.visual_nn = VisualNN(model, parent=self)
        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setWidget(self.visual_nn)
        self.setGeometry(0, 0, 1000, 1000)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.scroll.setGeometry(0, 0, self.width(), self.height())

