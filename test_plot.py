from PyQt6 import QtWidgets, QtCore
import pyqtgraph.opengl as gl
import numpy as np
import sys


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Layout
        layout = QtWidgets.QVBoxLayout(self)

        # 3D plot widget
        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)

        # Add grid
        grid = gl.GLGridItem()
        self.view.addItem(grid)

        # Sample data
        x, y, z = np.random.normal(size=(3, 1000))
        self.pos = np.vstack([x, y, z]).T

        self.scatter = gl.GLScatterPlotItem(pos=self.pos, color=(1, 0, 0, 1), size=5)
        self.view.addItem(self.scatter)

        # Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(20)
        self.slider.setValue(5)
        self.slider.valueChanged.connect(self.update_size)
        layout.addWidget(self.slider)

        self.setWindowTitle("3D Plot with Slider (PyQt6)")
        self.resize(800, 600)

    def update_size(self, value):
        self.scatter.setData(pos=self.pos, size=value)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())