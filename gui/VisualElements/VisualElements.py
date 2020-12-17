from .Interactive import *
from PyQt5 import QtWidgets, QtGui, QtCore
from ml import *
from .Layout import Layout
from .ScrollArea import ScrollArea
import random
from .Connection import CONNECTIONS


VISUAL_ELEMENTS = {}


def getWidgetFromId(__id):
    return VISUAL_ELEMENTS[__id]


class VisualElement(Interactive):
    def __init__(self, name='', parent=None, x=None, y=None, ID=None):
        super(VisualElement, self).__init__(parent)
        if ID is None: self.id = id(self)
        else: self.id = ID
        self.value = None
        VISUAL_ELEMENTS[self.id] = self
        self.label = QtWidgets.QLabel(name)
        self.label.setStyleSheet(f"color: white; "
                                 f"font-size: {self.get_size(50)}px")
        self.label.setFixedHeight(self.get_size(50))
        self.content.addWidget(self.label)
        if x and y: self.move(x, y)
        self.value = None

    def draw(self):
        self.alpha = 0.3
        self.color1 = QtGui.QColor("#963d97")
        self.color2 = QtGui.QColor("#e03a3c")
        rect = QtCore.QRect(0, 0, self.width(), self.height())
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0, self.color1)
        gradient.setColorAt(self.alpha, self.color2)
        self.painter.fillRect(rect, gradient)

    def build(self):
        self.value = self.init()

    def init(self):
        pass


class VisualDense(VisualElement):
    def __init__(self, parent=None, x=None, y=None, ID=None):
        super(VisualDense, self).__init__("Dense", parent, x, y, ID)
        self.setGeometry(self.x(), self.y(), 500, 400)
        self.input_dim_edit = QtWidgets.QLineEdit()
        self.output_dim_edit = QtWidgets.QLineEdit()
        self.contains_bias_edit = QtWidgets.QCheckBox()
        self.input_dim_label = QtWidgets.QLabel("input_dim")
        self.output_dim_label = QtWidgets.QLabel("output_dim")
        self.contains_bias_label = QtWidgets.QLabel("contains_bias")

        self.input_dim_label.setStyleSheet(
            f"color: white; font-size: {self.get_size(30)}px")
        self.output_dim_label.setStyleSheet(
            f"color: white; font-size: {self.get_size(30)}px")
        self.contains_bias_label.setStyleSheet(
            f"color: white; font-size: {self.get_size(30)}px")

        self.input_dim = Layout('H')
        self.output_dim = Layout('H')
        self.contains_bias = Layout('H')

        self.input_dim.addWidgets(self.input_dim_label, self.input_dim_edit)
        self.output_dim.addWidgets(self.output_dim_label, self.output_dim_edit)
        self.contains_bias.addWidgets(self.contains_bias_label, self.contains_bias_edit)

        self.content.addWidget(self.input_dim)
        self.content.addWidget(self.output_dim)
        self.content.addWidget(self.contains_bias)

    def init(self):
        input_dim = int(self.input_dim_edit.text())
        output_dim = int(self.output_dim_edit.text())
        contains_bias = self.contains_bias_edit.checkState()
        if contains_bias == QtCore.Qt.Checked:
            contains_bias = True
        else: contains_bias = False
        self.value = Dense(input_dim, output_dim, contains_bias=contains_bias)
        return self.value

    def __call__(self, *args, **kwargs):
        output = self.init()(*args, **kwargs)
        if self in CONNECTIONS.keys():
            connections = CONNECTIONS[self]
            for connection in connections:
                connection.__call__(output)


class VisualInput(VisualElement):
    def __init__(self, parent=None, x=None, y=None, ID=None):
        super(VisualInput, self).__init__("Input", parent, x, y, ID)
        self.setGeometry(self.x(), self.y(), 300, 200)
        self.alpha = 1

    def init(self):
        pass

    def __call__(self, *args, **kwargs):
        print(CONNECTIONS)
        if self in CONNECTIONS.keys():
            connections = CONNECTIONS[self]
            for connection in connections:
                connection.__call__()


class VisualSigmoid(VisualElement):
    def __init__(self, parent=None, x=None, y=None, ID=None):
        super(VisualSigmoid, self).__init__("Sigmoid", parent, x, y, ID)
        self.setGeometry(self.x(), self.y(), 300, 200)
        self.alpha = 1

    def init(self):
        self.value = Sigmoid()
        return self.value

    def __call__(self, *args, **kwargs):
        output = self.init()(*args, **kwargs)
        if self in CONNECTIONS.keys():
            connections = CONNECTIONS[self]
            for connection in connections:
                connection.__call__(output)


class VisualTanh(VisualElement):
    def __init__(self, parent=None, x=None, y=None, ID=None):
        super(VisualTanh, self).__init__("Tanh", parent, x, y, ID)
        self.setGeometry(0, 0, 200, 200)
        self.alpha = 1

    def init(self):
        self.value = Tanh()
        return self.value


class VisualReLU(VisualElement):
    def __init__(self, parent=None, x=None, y=None):
        super(VisualReLU, self).__init__("ReLU", parent, x, y)
        self.setGeometry(0, 0, 200, 200)
        self.alpha = 1

    def init(self):
        self.value = ReLU()
        return self.value


class VisualVector(VisualElement):
    def __init__(self, parent=None, x=None, y=None, ID=None):
        super(VisualVector, self).__init__("Vector", parent, x, y, ID)
        self.setGeometry(self.x(), self.y(), 300, 700)
        self.combobox = QtWidgets.QComboBox()
        self.combobox.addItems(["zeros", "random", "range"])
        self.combobox.activated[str].connect(self.change_mode)
        self.edit = QtWidgets.QLineEdit()
        self.edit.textChanged.connect(self.change_length)
        self.scroll = ScrollArea()
        self.scroll.resize(self.size())
        self.scroll.setFixedHeight(self.height() - 100)
        self.scroll.setFixedWidth(self.width() - 50)

        self.content.addWidget(self.combobox)
        self.content.addWidget(self.edit)
        self.content.addWidget(self.scroll)
        self.generator = lambda x: 0

    def change_length(self):
        text = self.edit.text()
        try:
            length = int(text)
            self.scroll.clear()
            for i in range(length):
                string = str(self.generator(i))
                if len(string) > 10:
                    string = string[:10]
                self.scroll.addWidget(
                    QtWidgets.QLineEdit(string))
        except ValueError: pass

    def change_mode(self, mode):
        if mode == "zeros":
            self.generator = lambda x: 0
        elif mode == "random":
            self.generator = lambda x: random.random()
        elif mode == "range":
            self.generator = lambda x: x

    def init(self):
        data = []
        layout = self.scroll.layout()
        for i in range(layout.count()):
            text = layout.itemAt(i).widget().text()
            data.append(float(text))
        self.value = Tensor(ndarray(data, shape=(1, len(data))))
        return self.value

    def __call__(self, x: Tensor = None):
        if x is not None:
            data = x.get_data()
            self.scroll.clear()
            for i in range(len(data)):
                string = str(data[i])
                if len(string) > 10:
                    string = string[:10]
                self.scroll.addWidget(
                    QtWidgets.QLineEdit(string))
        else:
            output = self.init()
            if self in CONNECTIONS.keys():
                connections = CONNECTIONS[self]
                for connection in connections:
                    connection.__call__(output)


DICTIONARY = {
    "Dense": VisualDense,
    "Sigmoid": VisualSigmoid,
    "Tanh": VisualTanh,
    "ReLU": VisualReLU,
    "Vector": VisualVector,
    "Input": VisualInput,
    VisualDense: Dense,
    VisualSigmoid: Sigmoid,
    VisualVector: Tensor,
}
