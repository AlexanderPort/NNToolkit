from .Interactive import *
from .ScrollArea import ScrollArea
from .VisualElements import *
from .Connection import *


PARAMETERS = {
    "Layers": ["Input", "Dense", "Conv2D", "RNN cell", "LSTM cell", "GRU cell"],
    "Activations": ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "Softmax"],
    "Optimizers": ["Optimizer", "SGD", "Momentum", "RMSprop", "Adam"],
    "Regularization": ["L2 norm", "Dropout", "Batch normalization"],
    "Structures": ["Scalar", "Vector", "Matrix", "Tensor"]
}

AREA = [None]
VISUAL_ELEMENT = None


class ClickableLabel(QtWidgets.QLabel):
    def __init__(self, text='', parent=None):
        super(ClickableLabel, self).__init__(text=text, parent=parent)
        self.parent = parent
        self.font_size = 30
        self.setStyleSheet(f"font-size: {self.get_size(self.font_size)}px")

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        global VISUAL_ELEMENT
        VISUAL_ELEMENT = (self.text(), self.pos())
        if AREA[0] is not None: AREA[0].update()

    def get_size(self, size):
        return size * SCALE


class Panel(Interactive):
    def __init__(self, parent=None):
        super(Panel, self).__init__(parent)
        self.setGeometry(300, 300, 400, 500)
        self.combobox = QtWidgets.QComboBox(self)
        self.combobox.resize(self.width(), self.combobox.height() * 2)
        self.combobox.addItems(PARAMETERS.keys())
        self.combobox.activated[str].connect(self.activated)
        self.font_size = 30
        self.combobox.setStyleSheet(f"background-color: #963d97;"
                                    f"color: white;"
                                    f"font-size: {self.get_size(self.font_size)}px")

        self.scroll = ScrollArea(self)
        self.scroll.setParent(self)
        self.scroll.move(0, self.combobox.height())
        self.scroll.resize(self.width(), self.height() -
                           self.combobox.height())
        self.scroll.setStyleSheet(f"background-color: #e03a3c;"
                                  f"color: white;"
                                  f"font-size: {self.get_size(self.font_size)}px")
        self.__update__(list(PARAMETERS.keys())[0])

    def __update__(self, key):
        self.scroll.clear()
        parameters = PARAMETERS[key]
        for i in range(len(parameters)):
            label = ClickableLabel(parameters[i])
            label.parent = self.parent()
            self.scroll.addWidget(label)

    def activated(self, key):
        self.__update__(key)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        pass
        # self.scroll.setGeometry(self.geometry())

    def addWidget(self, widget):
        self.widgets.append(widget)
        for widget in self.widgets:
            widget.resize(self.width(), self.height())
        layout = self.content.content()
        layout.insertWidget(layout.count() - 1, widget)


class Area(QtWidgets.QWidget, Connectable):
    def __init__(self, parent=None, __input=None):
        super(Area, self).__init__(parent)
        self.setGeometry(300, 300, 20000, 30000)
        if isinstance(__input, VisualElement):
            __input.setParent(self)
            self.visual_elements = [__input]
        else: self.visual_elements = []
        self.connections = []
        self.pos3 = QtCore.QPoint(0, 0)

    def update(self) -> None:
        global VISUAL_ELEMENT
        if VISUAL_ELEMENT:
            try:
                visual_element = DICTIONARY[VISUAL_ELEMENT[0]](self)
                visual_element.move(VISUAL_ELEMENT[1].x(),
                                    VISUAL_ELEMENT[1].y())
                self.visual_elements.append(visual_element)
                visual_element.show()
                VISUAL_ELEMENT = None
            except KeyError:
                pass

        for connection in self.connections:
            connection.update()
        super(Area, self).update()

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key_Delete:
            visual_elements = []
            for i in range(len(self.visual_elements)):
                element = self.visual_elements[i]
                if not element.active:
                    visual_elements.append(element)
                else:
                    if element in ELEMENT_TO_CONNECTION.keys():
                        connections = ELEMENT_TO_CONNECTION.pop(element)
                        for connection in connections:
                            if connection in self.connections:
                                self.connections.remove(connection)
                                connection.deleteLater()
                    if element in CONNECTIONS.keys():
                        CONNECTIONS.pop(element)
                    if element in PARENTS.keys():
                        for parent in PARENTS[element]:
                            if parent in CONNECTIONS.keys():
                                if element in CONNECTIONS[parent]:
                                    CONNECTIONS[parent].remove(element)
                    element.deleteLater()
            self.visual_elements = visual_elements

            connections = []
            for i in range(len(self.connections)):
                connection = self.connections[i]
                if not connection.active:
                    connections.append(connection)
                else:
                    connect_from = connection.connect_from
                    connect_to = connection.connect_to
                    if connect_from in CONNECTIONS.keys():
                        if connect_to in CONNECTIONS[connect_from]:
                            CONNECTIONS[connect_from].remove(connect_to)
                        ELEMENT_TO_CONNECTION[connect_from].remove(connection)
                        ELEMENT_TO_CONNECTION[connect_to].remove(connection)
                    connection.deleteLater()
            self.connections = connections

