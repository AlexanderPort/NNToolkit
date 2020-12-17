from PyQt5 import QtWidgets, QtGui


class Image(QtWidgets.QLabel):
    def __init__(self, path, parent=None):
        super(Image, self).__init__(parent=parent)
        self.pixmap = QtGui.QPixmap(path)
        self.setPixmap(self.pixmap)
        self.resize(self.pixmap.width(), self.pixmap.height())