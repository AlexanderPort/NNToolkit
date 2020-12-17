from PyQt5 import QtWidgets, QtCore, QtGui


class ScrollArea(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ScrollArea, self).__init__(parent=parent)
        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        self.content.setLayout(layout)
        self.scroll.setWidget(self.content)

    def setVerticalScrollBarPolicy(self, policy):
        self.scroll.setVerticalScrollBarPolicy(policy)

    def setHorizontalScrollBarPolicy(self, policy):
        self.scroll.setHorizontalScrollBarPolicy(policy)

    def addWidget(self, widget: QtWidgets.QWidget):
        self.content.layout().addWidget(widget)

    def clear(self):
        layout = self.content.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().deleteLater()
        print(layout.count())

    def move(self, *args, **kwargs):
        self.scroll.move(*args, **kwargs)

    def resize(self, *args, **kwargs):
        self.scroll.resize(*args, **kwargs)

    def setParent(self, parent: QtWidgets.QWidget) -> None:
        self.scroll.setParent(parent)

    def setStyleSheet(self, styleSheet: str) -> None:
        self.content.setStyleSheet(styleSheet)

    def layout(self):
        return self.content.layout()