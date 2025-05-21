from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
import time
    
def check_if_valid(val):
    try:
        float(val)
        return True
    except:
        return False
    
SPINBOX_STYLESHEET = """
    QDoubleSpinBox {
        border: 1px solid #B0B0B0;  /* Border color */
        border-radius: 3px;         /* Rounded corners */
        padding: 3px;                /* Padding inside the box */
    }

    QDoubleSpinBox:focus {
        border-color: #5F9EA0;      /* Border color when focused */
    }
    
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        width: 0;
        height: 0;
        border: none;
    }
"""


class PointInputRow(QtWidgets.QWidget):
    """A single row of x, y, z, time inputs."""
    values_changed = QtCore.pyqtSignal()
    delete_clicked = QtCore.pyqtSignal()

    def __init__(self, id=None):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        self.x_input = QtWidgets.QDoubleSpinBox()
        self.y_input = QtWidgets.QDoubleSpinBox()
        self.z_input = QtWidgets.QDoubleSpinBox()
        self.t_input = QtWidgets.QDoubleSpinBox()

        self.delete_button = QtWidgets.QPushButton("🗑️")  # Trash can button
        self.delete_button.setFixedWidth(25)
        self.delete_button.setStyleSheet('text-align: left;')
        
        if not id:
            self.row_id = str(time.time())  # in PointInputRow
        else:
            self.row_id = id

        for widget, label in zip(
            [self.x_input, self.y_input, self.z_input, self.t_input],
            ['x (mm)', 'y (mm)', 'z (um)', 't (ns)']
        ):
            # widget.setPlaceholderText(label)
            widget.setMinimumWidth(60)
            if label not in ["z (um)", "t (ns)"]:
                widget.setDecimals(4)
            else:
                widget.setDecimals(1)
            widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            widget.setStyleSheet(SPINBOX_STYLESHEET)
            widget.textChanged.connect(self.values_changed.emit)

            layout.addWidget(widget)
            
        layout.addStretch()
        layout.addWidget(self.delete_button)
        self.delete_button.clicked.connect(self.delete_clicked.emit)

    def get_values(self):
        values = tuple(float(v) if check_if_valid(v) else None for v in [self.x_input.text(), self.y_input.text(), self.z_input.text(), self.t_input.text()]) # roll one spot here to handle the axial rotation of the camera angle
        if all(v is None for v in values):
            return None
        else:
            return tuple(v if v is not None else 0 for v in values)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left: Input form
        self.form_widget = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QVBoxLayout(self.form_widget)

        self.add_row_button = QtWidgets.QPushButton("Add Row")
        self.add_row_button.clicked.connect(self.add_input_row)

        # self.plot_button = QtWidgets.QPushButton("Plot Points")
        # self.plot_button.clicked.connect(self.plot_points)
        velocity_row = QtWidgets.QHBoxLayout()
        velocity_label = QtWidgets.QLabel("Velocity (km/s): ")

        self.velocity_input = QtWidgets.QDoubleSpinBox()
        self.velocity_input.setRange(0, 7)
        self.velocity_input.setValue(5)
        self.velocity_input.setDecimals(2)
        self.velocity_input.setFixedWidth(80)
        
        self.velocity_input.setStyleSheet(SPINBOX_STYLESHEET)
        
        velocity_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        velocity_row.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        velocity_row.setStretch(0, 0)  # For the label
        velocity_row.setStretch(1, 0)  # For the spinbox
        velocity_row.setContentsMargins(11, 0, 0, 0)
        # velocity_row.setSpacing(5)
        velocity_row.addWidget(velocity_label)
        velocity_row.addWidget(self.velocity_input)
        self.form_layout.addLayout(velocity_row)
        
        coordinate_row = QtWidgets.QHBoxLayout()
        labels = ["x (mm)", "y (mm)", "z (um)", "t (ns)", "Delete"]
        for label in labels:
            widget = QtWidgets.QLabel(label)
            # widget.setContentsMargins(11, 0, 0, 0)
            if label == "Delete":
                widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            else:
                widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
            
            if label not in ["t (ns)", "Delete"]:
                widget.setFixedWidth(60)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            coordinate_row.addWidget(widget)

        coordinate_row.setContentsMargins(11, 10, 0, 0)
        self.form_layout.addLayout(coordinate_row)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.add_row_button)
        # button_row.addWidget(self.plot_button)
        self.form_layout.addLayout(button_row)
        self.form_layout.addStretch()

        # Right: 3D plot
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=3e-2, elevation=0, azimuth=75) # elevation and azimuth set viewing angle
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        # self.grid = gl.GLGridItem()
        # self.view.addItem(self.grid)
        # Add grid items for all planes (X-Y, X-Z, Y-Z)

        self.grid_xy = gl.GLGridItem()
        self.grid_xy.setSize(2e-2, 2e-2)  # 10 cm grid
        self.grid_xy.setSpacing(1e-3, 1e-3)    # 1 mm density
        self.view.addItem(self.grid_xy)
        self.grid_xy.rotate(90, 0, 1, 0)  

        # Initial scatter plot (empty)
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), color=(0, 0, 0, 1), size=5)
        self.view.addItem(self.scatter)
        
        # Right side layout for plot + slider
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.view)

        # Create and configure slider
        self.slider_row = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(-1000)
        self.slider.setMaximum(1000)
        self.slider.setValue(0)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(50)

        self.slider_value = QtWidgets.QLabel(f"Time: {self.slider.value()} ns")
        self.slider_value.setFixedWidth(80)
        
        self.slider_row.addWidget(self.slider_value)
        self.slider_row.addWidget(self.slider)
        
        slider_widget = QtWidgets.QWidget()
        slider_widget.setLayout(self.slider_row)

        right_layout.addWidget(slider_widget)

        # Connect slider to update
        self.slider.valueChanged.connect(self.on_slider_changed)

        if not hasattr(self, "plane"):
            self.make_circular_plane()

        # Use a container widget to add to the main layout
        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_layout)

        # Combine left and right in main layout
        main_layout.addWidget(self.form_widget, stretch=2)
        main_layout.addWidget(right_container, stretch=5)

        self.setWindowTitle("Impact visualization tool (1000x zoom along impact axis)")
        self.resize(1200, 600)

        self.settings = QtCore.QSettings("DCS", "Impact-Visualizer")
        self.restoreGeometry(self.settings.value("geometry", b""))
        self.restoreGeometry(self.settings.value("windowState", b""))
        
        row_keys = [k for k in self.settings.allKeys() if k.startswith("row_")]
        row_keys.sort()
        self.input_rows = []

        for i, key in enumerate(row_keys):
            values = self.settings.value(key)
            self.add_input_row(id=key.split("_")[-1])
            if values:
                x, y, z, t = (float(v) if check_if_valid(v) else 0 for v in values)
                row = self.input_rows[-1]
                row.x_input.setValue(self.parse_val(x))
                row.y_input.setValue(self.parse_val(y))
                row.z_input.setValue(self.parse_val(z))
                row.t_input.setValue(self.parse_val(t))

        if len(self.input_rows) == 0:
            self.add_input_row()
            

    def on_slider_changed(self, val):
        self.slider_value.setText(f"Time: {val} ns")
        self.move_plane(t=val)
        self.plot_points()
        

    def move_plane(self, t):
        m = QtGui.QMatrix4x4()
        m.translate(-self.velocity_input.value()*1e3 * t*1e-9, 0, 0)
        # m.rotate(self.velocity_input.value() * t, 0, 0, 1)
        self.plane.setTransform(m)
        self.center = self.plane.transform().map(QtGui.QVector3D(0, 0, 0))
        self.z = self.center.x()
        

    def make_circular_plane(self, radius=10e-3, segments=60):
        z = 0
        vertices = [[z, 0, 0]]

        # Circle perimeter points
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([z, x, y])

        vertices = np.array(vertices, dtype=np.float32)

        # Triangulate: connect center point to each edge segment
        faces = []
        for i in range(1, segments + 1):
            faces.append([i+1, i, 0])
        faces = np.array(faces)

        # Create the mesh
        self.plane = gl.GLMeshItem(vertexes=vertices, faces=faces, color=(0.3, 0.6, 1.0, 0.3), smooth=True, drawEdges=False)
        self.plane.setGLOptions('translucent')
        self.view.addItem(self.plane)
    
                
    def parse_val(self, val):
        """ return int if int, if has float val, return str() of float"""
        return val
        # if int(val) == float(val):
        #     return str(int(val))
        # else:
        #     return str(val)


    def add_input_row(self, id=None):
        row = PointInputRow(id=id)
        row.values_changed.connect(self.plot_points)
        row.delete_clicked.connect(self.delete_input_row)
        self.input_rows.append(row)
        self.form_layout.insertWidget(self.form_layout.count() - 2, row)  # Add above buttons
        
    def delete_input_row(self):
        row = self.sender()
        index = self.form_layout.indexOf(row)
        
        if index != -1:
            self.input_rows.remove(row)
            row.deleteLater()
            self.settings.remove(f"row_{row.row_id}")

        self.plot_points()

    def plot_points(self):
        points = []
        for row in self.input_rows:
            values = row.get_values()

            if values:
                x, y, z, t = values
                points.append(np.roll(np.array((x * 1e-3, y * 1e-3, z * 1e-3)), 1))
                
        if points:
            pos = np.array(points)
            if not hasattr(self, "z"):
                self.z = 0

            colors = []
            for row in pos:
                if row[0] > self.z:
                    colors.append((0, 1, 0, 1))
                else:
                    colors.append((1, 0, 0, 1))

            self.scatter.setData(pos=pos, color=np.array(colors))

        else:
            self.scatter.setData(pos=np.zeros((0, 3)), color=(0, 0, 0, 1))
            
    def closeEvent(self, event):
        self.settings = QtCore.QSettings("DCS", "Impact-Visualizer")
        for i, row in enumerate(self.input_rows):
            values = row.get_values()
            self.settings.setValue(f"row_{row.row_id}", values)

        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
