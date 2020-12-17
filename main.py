from PyQt5 import QtWidgets


if __name__ == '__main__':
    import sys
    from gui import ScrollableMainWindow
    app = QtWidgets.QApplication(sys.argv)
    nn = ScrollableMainWindow()
    nn.show()
    sys.exit(app.exec_())
