from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5 import QtCore
import sys
from .VisualElements.VisualElements import *
from .VisualElements.Area import Area, Panel, AREA
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QAction, QApplication
from PyQt5.QtGui import QIcon
from .VisualNN import *

PARAMETERS = {
    "Layers": ["Input", "Dense", "Conv2D", "RNN cell", "LSTM cell", "GRU cell"],
    "Activations": ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "Softmax"],
    "Optimizers": ["Optimizer", "SGD", "Momentum", "RMSprop", "Adam"],
    "Regularization": ["L2 norm", "Dropout", "Batch normalization"],
    "Structures": ["Scalar", "Vector", "Matrix", "Tensor"]
}

INPUT = None
VISUAL_ELEMENT = None


class MainMenu(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainMenu, self).__init__(parent)
        newAction = QAction('New', parent)
        newAction.triggered.connect(self.new)

        exitAction = QAction('Exit', parent)
        exitAction.triggered.connect(self.close)

        saveAction = QAction('Save', parent)
        saveAction.triggered.connect(self.save)

        openAction = QAction('Open', parent)
        openAction.triggered.connect(self.open)

        buildAction = QAction('Build', parent)
        buildAction.triggered.connect(self.build)

        runAction = QAction('Run', parent)
        runAction.triggered.connect(self.run)

        menubar = parent.menuBar()

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(newAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(exitAction)
        fileMenu.resize(200, 200)

        buildMenu = menubar.addMenu('&Build')
        buildMenu.addAction(buildAction)
        buildMenu.resize(200, 200)

        runMenu = menubar.addMenu('&Run')
        runMenu.addAction(runAction)
        runMenu.resize(200, 200)

        self.visualizeMenu = menubar.addMenu('&Visualize')
        self.visualizeAction = self.visualizeMenu.addAction('Visualize')
        self.visualizeAction.triggered.connect(self.visualize)
        self.visualizeMenu.resize(200, 200)

        self.table_widget = self.parent().table_widget

        '''
        text, ok = QtWidgets.QInputDialog.getText(self, 'Installing C extension',
                                                  'Enter your Python interpreter path')
        
        if ok:
            try:
                import cndarray
            except ImportError:
                try:
                    import os
                    os.system('python setup.py install')
                except:
                    pass
        '''

    def save(self):
        tab = self.table_widget.tab
        area = self.table_widget.areas[tab]
        visual_elements = area.visual_elements
        connections = area.connections
        text = ''
        element: VisualElement
        connection: Connection
        for element in visual_elements:
            text += f'{element.__class__.__name__}(x={element.x()}, ' \
                    f'y={element.y()}, ID={element.id})\n'
        for connection in connections:
            text += f'{connection.__class__.__name__}' \
                    f'(connect_from=getWidgetFromId({connection.connect_from.id}), ' \
                    f'connect_to=getWidgetFromId({connection.connect_to.id}))\n'
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self)
        with open(fname + ".project", mode='w', encoding='utf-8') as file:
            file.write(text)

    def open(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self)
        file = open(fname, mode='r', encoding='utf-8')
        lines = file.readlines()
        tab = QtWidgets.QWidget()
        self.table_widget.addTab(tab, fname)
        area = self.table_widget.areas[tab]
        for line in lines:
            widget: QtWidgets.QWidget = eval(line)
            widget.setParent(area)
            if isinstance(widget, VisualElement):
                if isinstance(widget, VisualInput):
                    input_ = self.table_widget.inputs[tab]
                    area.visual_elements.remove(input_)
                    input_.deleteLater()
                    self.table_widget.inputs[tab] = widget
                area.visual_elements.append(widget)
                widget.active = False
            elif isinstance(widget, Connection):
                area.connections.append(widget)
            widget.show()

    def build(self):
        tab = self.table_widget.tab
        area = self.table_widget.areas[tab]
        for element in area.visual_elements:
            element.build()

    def run(self):
        if INPUT is not None: INPUT()

    def new(self):
        self.parent().table_widget.addTab()

    def get_models(self, widget, model, models):
        types = [VisualDense, VisualSigmoid]
        if model not in models: models.append(model)
        if type(widget) in types:
            print(widget)
            if widget.value is not None:
                model.addLayer(widget.value)
            else: model.addLayer(widget.init())
        if widget in CONNECTIONS.keys():
            connections = CONNECTIONS[widget]
            for connection in connections:
                self.get_models(connection, model.copy(), models)

    def visualize(self, *args):
        models = []
        model = Sequential([])
        print('_______________________________________________________')
        self.get_models(INPUT, model, models)
        print(models)
        print(models[-1].layers)
        self.nn = ScrollableVisualNN(models[-1])
        self.nn.show()
        

class TableWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(TableWidget, self).__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)

        self.tab: QtWidgets.QWidget = None
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self.change)
        self.tabs_dict = {}
        self.tabs_list = []
        self.areas = {}
        self.inputs = {}
        self.connections = {}
        self.tabs.resize(300, 200)

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def addTab(self, tab=None, name='untitled'):
        if tab is None: tab = QtWidgets.QWidget()
        self.inputs[tab] = VisualInput(x=100, y=100)
        self.areas[tab] = Area(tab, self.inputs[tab])
        self.connections[tab] = []
        self.tabs.addTab(tab, name)
        self.tabs_dict[name] = tab
        self.tabs_list.append(tab)

    def change(self, index):
        if len(self.tabs_list) != 0:
            self.tab = self.tabs_list[index]
            global AREA, CONNECTIONS, INPUT
            INPUT = self.inputs[self.tab]
            AREA[0] = self.areas[self.tab]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setGeometry(0, 0, 3000, 3000)
        self.painter = QtGui.QPainter(self)
        self.table_widget = TableWidget(self)
        self.setCentralWidget(self.table_widget)
        self.table_widget.setFixedWidth(self.width())

        self.initUI()

    def initUI(self):
        self.menu = MainMenu(self)
        self.panel = Panel(self)
        self.panel.move(50, 50)
        self.tab = QtWidgets.QWidget()
        self.table_widget.addTab(self.tab)
        global AREA, CONNECTIONS, INPUT
        self.table_widget.tab = self.tab
        INPUT = self.table_widget.inputs[self.tab]
        AREA[0] = self.table_widget.areas[self.tab]

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if AREA[0] is not None: AREA[0].keyPressEvent(a0)


class ScrollableMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ScrollableMainWindow, self).__init__()
        self.window = MainWindow(self)
        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setWidget(self.window)
        self.setGeometry(0, 0, 2000, 2000)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.scroll.setGeometry(0, 0, self.width(), self.height())

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        self.window.keyPressEvent(a0)