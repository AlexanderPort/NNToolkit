from PyQt5 import QtWidgets


class Layout(QtWidgets.QWidget):
    def __init__(self, orientation=None, parent=None):
        super(Layout, self).__init__(parent)
        self.orientation = orientation
        self.change_orientation(orientation)

    def addWidget(self, widget: QtWidgets.QWidget):
        if self.layout() is not None:
            self.layout().addWidget(widget)
        else: widget.setParent(self)

    def addWidgets(self, *widgets):
        for widget in widgets:
            self.addWidget(widget)

    def change_orientation(self, orientation):
        self.orientation = orientation
        if orientation in ('H', 'h'):
            self.setLayout(QtWidgets.QHBoxLayout())
        elif orientation in ('V', 'v'):
            self.setLayout(QtWidgets.QVBoxLayout())
        elif orientation is None: pass
        else:
            raise Exception('such orientation is not exist')
