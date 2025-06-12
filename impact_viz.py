import traceback

try:
    import sys
    import time

    import numpy as np
    import pyqtgraph.opengl as gl
    from PyQt6 import QtCore, QtGui, QtWidgets
    from calculate_tilt_robust import Tilt


    def check_if_valid(val: str) -> bool:
        try:
            float(val)
        except TypeError:
            return False
        else:
            return True


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

    COMBO_STYLESHEET = """
    QComboBox {
        padding-right: 20px;  /* Space for the arrow */
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid gray;
    }

    QComboBox::down-arrow {
        image: url(:/qt-project.org/styles/commonstyle/images/arrowdown-16.png);
    }
    """


    class PointInputRow(QtWidgets.QWidget):
        """A single row of x, y, z, time inputs."""

        values_changed = QtCore.pyqtSignal()
        delete_clicked = QtCore.pyqtSignal()

        def __init__(self, id=None) -> None:  # noqa: A002
            super().__init__()
            layout = QtWidgets.QHBoxLayout(self)
            self.x_input = QtWidgets.QDoubleSpinBox()
            self.y_input = QtWidgets.QDoubleSpinBox()
            self.z_input = QtWidgets.QDoubleSpinBox()
            self.t_input = QtWidgets.QDoubleSpinBox()

            self.delete_button = QtWidgets.QPushButton("🗑️")  # Trash can button
            self.delete_button.setFixedWidth(25)
            self.delete_button.setStyleSheet("text-align: left;")

            if not id:
                self.row_id = str(time.time())  # in PointInputRow
            else:
                self.row_id = id

            for widget, label in zip(
                [self.x_input, self.y_input, self.z_input, self.t_input],
                ["x (mm)", "y (mm)", "z (um)", "t (ns)"],
            ):
                # widget.setPlaceholderText(label)
                widget.setMinimumWidth(65)
                if label == "z (um)":
                    widget.setDecimals(1)
                    widget.setRange(-50, 50)
                elif label == "t (ns)":
                    widget.setDecimals(2)
                    widget.setRange(-200, 200)
                else:
                    widget.setDecimals(4)
                    widget.setRange(-10, 10)
                widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                widget.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )
                widget.setStyleSheet(SPINBOX_STYLESHEET)
                widget.textChanged.connect(self.values_changed.emit)

                layout.addWidget(widget)

            layout.addStretch()
            layout.addWidget(self.delete_button)
            self.delete_button.clicked.connect(self.delete_clicked.emit)

        def get_values(self):
            values = tuple(
                float(v) if check_if_valid(v) else None
                for v in [
                    self.x_input.text(),
                    self.y_input.text(),
                    self.z_input.text(),
                    self.t_input.text(),
                ]
            )  # roll one spot here to handle the axial rotation of the camera angle
            if all(v is None for v in values):
                return None
            return tuple(v if v is not None else 0 for v in values)


    class MainWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            main_layout = QtWidgets.QHBoxLayout(self)
            self.matrix = QtGui.QMatrix4x4()

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
            self.velocity_input.setDecimals(3)
            self.velocity_input.setFixedWidth(80)

            self.velocity_input.setStyleSheet(SPINBOX_STYLESHEET)
            
            mode_label = QtWidgets.QLabel("Mode: ")
            self.color_method = QtWidgets.QComboBox()
            self.color_method.addItems(["Timing", "Location"])
            self.color_method.currentTextChanged.connect(self.plot_points)
            # self.color_method.setStyleSheet(COMBO_STYLESHEET)

            velocity_label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            velocity_row.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

            velocity_row.setStretch(0, 0)  # For the label
            velocity_row.setStretch(1, 0)  # For the spinbox
            velocity_row.setContentsMargins(11, 0, 0, 0)
            # velocity_row.setSpacing(5)
            velocity_row.addWidget(velocity_label)
            velocity_row.addWidget(self.velocity_input)
            velocity_row.addWidget(mode_label)
            velocity_row.addWidget(self.color_method)
            self.form_layout.addLayout(velocity_row)

            magnification_row = QtWidgets.QHBoxLayout()
            self.magnification = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.magnification.setMinimum(0)
            self.magnification.setMaximum(3)
            self.magnification.setValue(3)
            self.magnification.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            self.magnification.setTickInterval(1)

            self.magnification.valueChanged.connect(self.magnification_changed)

            self.mag_value = QtWidgets.QLabel(f"{10**self.magnification.value()}x")
            self.mag_value.setFixedWidth(90)

            magnification_row.addWidget(self.mag_value)
            magnification_row.addWidget(self.magnification)

            mag_widget = QtWidgets.QWidget()
            mag_widget.setLayout(magnification_row)

            self.form_layout.addWidget(mag_widget)
            
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            self.form_layout.addWidget(line)
            
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
                    widget.setFixedWidth(65)
                widget.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )
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
            self.view.setCameraPosition(
                distance=3e-2,
                elevation=0,
                azimuth=75,
            )  # elevation and azimuth set viewing angle
            self.view.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )

            self.gl_frame = QtWidgets.QFrame()
            self.gl_frame.setStyleSheet("""
                QFrame {
                    border: 1px solid #B0B0B0;  /* Border color */
                    border-radius: 5px;
                        }
            """)

            # Inner container to hold the GLViewWidget inside the frame
            inner_widget = QtWidgets.QWidget()
            inner_layout = QtWidgets.QVBoxLayout(inner_widget)
            inner_layout.setContentsMargins(2, 2, 2, 2)
            inner_layout.addWidget(self.view)

            # Layout for the frame
            frame_layout = QtWidgets.QVBoxLayout(self.gl_frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)
            frame_layout.addWidget(inner_widget)

            # self.grid = gl.GLGridItem()
            # self.view.addItem(self.grid)
            # Add grid items for all planes (X-Y, X-Z, Y-Z)

            self.grid_xy = gl.GLGridItem()
            self.grid_xy.setSize(2e-2, 2e-2)  # 10 cm grid
            self.grid_xy.setSpacing(1e-3, 1e-3)  # 1 mm density
            self.view.addItem(self.grid_xy)
            self.grid_xy.rotate(90, 0, 1, 0)

            # Initial scatter plot (empty)
            self.scatter = gl.GLScatterPlotItem(
                pos=np.zeros((0, 3)), color=(0, 0, 0, 1), size=5
            )
            self.view.addItem(self.scatter)

            # Right side layout for plot + slider
            right_layout = QtWidgets.QVBoxLayout()
            right_layout.addWidget(self.gl_frame)

            # Create and configure slider
            self.slider_row = QtWidgets.QHBoxLayout()
            self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.slider.setMinimum(-350)
            self.slider.setMaximum(350)
            self.slider.setValue(0)
            self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            # self.slider.setTickInterval(1)

            self.init_time = None
            self.time = self.slider.value() / 1e1
            # self.slider_value = QtWidgets.QLabel(f"Time: {self.time:.1f} ns")
            self.slider_value = QtWidgets.QLabel()
            self.set_time_text()
            self.slider_value.setFixedWidth(90)

            self.slider_row.addWidget(self.slider_value)
            self.slider_row.addWidget(self.slider)

            slider_widget = QtWidgets.QWidget()
            slider_widget.setLayout(self.slider_row)

            right_layout.addWidget(slider_widget)

            # Connect slider to update
            self.slider.valueChanged.connect(self.on_slider_changed)

            if not hasattr(self, "plane"):
                # magnification_ratios = {1:10e-3, 10:20e-3, 100:30e-3, 1000:40e-3}
                self.make_circular_plane(radius=(1+2*self.magnification.value())*10*1e-3)
                
            # TODO: sometimes gets stuck with new Timing mode, i think doesn't reset properly
            # TODO: input from .xlsx position/delay file

            # Use a container widget to add to the main layout
            right_container = QtWidgets.QWidget()
            right_container.setLayout(right_layout)

            # Combine left and right in main layout
            main_layout.addWidget(self.form_widget, stretch=2)
            main_layout.addWidget(right_container, stretch=5)

            self.resize(1200, 600)

            self.settings = QtCore.QSettings(
                QtCore.QSettings.Format.IniFormat,
                QtCore.QSettings.Scope.UserScope,
                "DCS",
                "Impact-Visualizer",
            )

            self.restoreGeometry(self.settings.value("geometry", b""))
            self.restoreGeometry(self.settings.value("windowState", b""))

            if not hasattr(self, "result"):
                self.result = QtWidgets.QLabel("")
                result_row = QtWidgets.QHBoxLayout()
                result_row.addWidget(self.result)
                self.form_layout.addLayout(result_row)
                

            row_keys = [k for k in self.settings.allKeys() if k.startswith("row_")]
            row_keys.sort()
            self.input_rows = []

            for i, key in enumerate(row_keys):
                values = self.settings.value(key)
                self.add_input_row(id=key.split("_")[-1])
                if values:
                    x, y, z, t = (float(v) if check_if_valid(v) else 0 for v in values)
                    row = self.input_rows[-1]
                    row.blockSignals(True)
                    row.x_input.setValue(self.parse_val(x))
                    row.y_input.setValue(self.parse_val(y))
                    row.z_input.setValue(self.parse_val(z))
                    row.t_input.setValue(self.parse_val(t))
                    row.blockSignals(False)
                
            if self.settings.value("magnification"):
                self.magnification.setValue(int(self.settings.value("magnification")))

            self.setWindowTitle(f"Impact visualization tool ({int(10**self.magnification.value())}x zoom along impact axis)")

            if len(self.input_rows) == 0:
                self.add_input_row()

            self.velocity_input.valueChanged.connect(self.plot_points)
            if self.settings.value("velocity", b""):
                self.velocity_input.blockSignals(True)
                self.velocity_input.setValue(float(self.settings.value("velocity")))
                self.velocity_input.blockSignals(False)
            if self.settings.value("time_slider", b""):
                # self.slider.blockSignals(True)
                self.slider.setValue(int(self.settings.value("time_slider")))
                # self.slider.blockSignals(False)
            else:
                self.slider.setValue(0)
            if self.settings.value("color_method", type=int) is not None:
                self.color_method.blockSignals(True)
                self.color_method.setCurrentIndex(int(self.settings.value("color_method")))
                self.color_method.blockSignals(False)

            # self.plot_points()
        
        def magnification_changed(self):
            # self.init_time = None
            self.setWindowTitle(f"Impact visualization tool ({int(10**self.magnification.value())}x zoom along impact axis)")
            self.mag_value.setText(f"{10**self.magnification.value()}x")
            self.modify_plane_size(radius=(1+2*self.magnification.value())*10*1e-3)
            self.plot_points()
            
        def set_time_text(self):
            if False:
            # if self.init_time is not None:
                text = f"Time: {self.time - self.init_time:.1f} ns"
            else:
                text = f"Time: {self.time:.1f} ns"
            self.slider_value.setText(text)

        def on_slider_changed(self, val):
            # if hasattr(self, "pins"):
            #     val -= np.median(self.pins[:, 3]) * 1e9
            self.time = val / 1e1
            self.set_time_text()
            self.move_plane(t=self.time)
            self.plot_points()

        def move_plane(self, t):
            self.z = -self.velocity_input.value() * 1e3 * t * 1e-9 * 10**self.magnification.value()
            self.matrix = QtGui.QMatrix4x4()
            self.matrix.translate(
                self.z, 0, 0
            )  # need to scale up to "zoom" in
            if hasattr(self, "magnified_normal_vecs_avg"):
                rotation_axis = np.cross(np.array([1,0,0]), np.roll(self.magnified_normal_vecs_avg, 1))
                self.matrix.rotate(-self.magnified_tilt_angle * 180 / np.pi, QtGui.QVector3D(*rotation_axis))
                # self.matrix.rotate(0, QtGui.QVector3D(*rotation_axis))

            self.plane.setTransform(self.matrix)
            
        def create_mesh_data(self, radius=10e-3, segments=60):
            z = 0
            vertices = [[z, 0, 0]]

            for i in range(segments + 1):
                angle = 2 * np.pi * i / segments
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                vertices.append([z, x, y])

            vertices = np.array(vertices, dtype=np.float32)

            # Triangulate: connect center point to each edge segment
            faces = []
            for i in range(1, segments + 1):
                faces.append([i + 1, i, 0])
            faces = np.array(faces)
            
            return gl.MeshData(vertexes=vertices, faces=faces)

        def make_circular_plane(self, radius=30e-3):
            # Create the mesh
            self.plane = gl.GLMeshItem(
                vertexes=[],
                faces=[],
                color=(0.3, 0.6, 1.0, 0.3),
                smooth=True,
                drawEdges=False,
            )

            # Circle perimeter points
            meshdata = self.create_mesh_data(radius=radius)
            self.plane.setMeshData(vertexes=meshdata.vertexes(), faces=meshdata.faces())

            self.plane.setGLOptions("translucent")
            self.view.addItem(self.plane)
            
        def modify_plane_size(self, radius=30e-3):
            meshdata = self.create_mesh_data(radius=radius)
            self.plane.setMeshData(vertexes=meshdata.vertexes(), faces=meshdata.faces())

        def parse_val(self, val):
            """return int if int, if has float val, return str() of float"""
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
            self.form_layout.insertWidget(
                self.form_layout.count() - 3, row
            )  # Add above buttons

        def delete_input_row(self):
            row = self.sender()
            index = self.form_layout.indexOf(row)

            if index != -1:
                self.input_rows.remove(row)
                row.deleteLater()
                self.settings.remove(f"row_{row.row_id}")

            self.plot_points()

        def plot_points(self, sender=None):
            points = []
            pins = []
            for row in self.input_rows:
                values = row.get_values()

                if values:
                    x, y, z, t = values
                    points.append(np.roll(np.array((x * 1e-3, y * 1e-3, z * 1e-6 * 10**self.magnification.value())), 1))
                    pins.append(np.array((x * 1e-3, y * 1e-3, z * 1e-6, t * 1e-9)))
                    
            normal = QtGui.QVector3D(
                self.matrix.column(0).x(),
                self.matrix.column(0).y(),
                self.matrix.column(0).z()
            ).normalized()
            point_on_plane = QtGui.QVector3D(
                self.matrix.column(3).x(),
                self.matrix.column(3).y(),
                self.matrix.column(3).z()
            )

            normal_vec = np.array([normal.x(), normal.y(), normal.z()])
            start = np.array([point_on_plane.x(), point_on_plane.y(), point_on_plane.z()])  
            end = start + normal_vec
            line_points = np.array([start, end])

            # if not hasattr(self, "normal"):
            #     self.normal = gl.GLLinePlotItem(pos=line_points)
            #     self.view.addItem(self.normal)
            
            # self.normal.setData(pos=line_points)

            if not hasattr(self, "pins"):
                self.pins = np.array(pins)
            elif not np.array_equal(np.array(pins), self.pins):
                self.pins = np.array(pins)
                self.init_time = None

            if points:
                pos = np.array(points)
                if not hasattr(self, "z"):
                    self.z = 0

                colors = []
                for idx, row in enumerate(pos):
                    v = QtGui.QVector3D(*row) - point_on_plane
                    d = QtGui.QVector3D.dotProduct(v, normal)
                    c = (1, 0, 0, 1)
                    if self.color_method.currentText() == "Location":
                        if d > 0:
                            c = (0, 1, 0, 1)
                    elif self.color_method.currentText() == "Timing":
                        # if self.init_time is None:
                        #     self.init_time = self.time
                        # else:
                        #     print(self.time, self.init_time, self.pins[idx, 3]*1e9)
                        #     if self.time > self.pins[idx, 3] * 1e9: # TODO: need to figure out a nice way to do the timing mode. maybe find the earliest time that a collision occurs and increment the timing from there? 
                        if self.init_time is None and self.pins[idx, 3] == np.min(self.pins[:, 3]) and d > 0:
                            self.init_time = np.copy(self.time)
                            self.set_time_text()
                        if self.init_time is not None and self.time - self.init_time >= (self.pins[idx, 3] - np.min(self.pins[:, 3])) * 1e9:
                            c = (0, 1, 0, 1)
                            
                        # print(self.init_time, self.time, self.pins[idx, 3], np.min(self.pins[:, 3]))

                    colors.append(c)

                self.scatter.setData(pos=pos, color=np.array(colors))

            else:
                self.scatter.setData(pos=np.zeros((0, 3)), color=(0, 0, 0, 1))

            self.calculate_angle()

        def calculate_angle(self):
            pins = self.pins
            tilt = Tilt(velocity=self.velocity_input.value(), pins=pins)
            angle, std, normal_vecs = tilt.iterate_tilt_calculation(
                save_data=False
                # save_data=True
            )
            _, _, magnified_normal_vecs = tilt.magnify_impact_axis(magnification=10**self.magnification.value()).iterate_tilt_calculation(save_data=False)

            additional_string = "" 
            if magnified_normal_vecs is None:
                magnified_normal_vecs = np.copy(normal_vecs)
                additional_string = " (magnification could not be shown)"
            
            try:
                avg_normalized = np.mean(normal_vecs, axis=0) / np.linalg.norm(np.mean(normal_vecs, axis=0))
                angle2 = np.arccos(avg_normalized[2])
                mean_resultant_length = np.linalg.norm(np.sum(normal_vecs, axis=0))/normal_vecs.shape[0]
                std2 = np.sqrt(-2 * np.log(mean_resultant_length)) # this gives something approximating the angular error, Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics. Wiley. ISBN: 978-0471953333 chap 9

                self.magnified_normal_vecs_avg = np.mean(magnified_normal_vecs, axis=0) / np.linalg.norm(np.mean(magnified_normal_vecs, axis=0))
                self.magnified_tilt_angle = np.arccos(self.magnified_normal_vecs_avg[2])
                
                if self.magnified_tilt_angle is not None:
                    self.move_plane(t=self.time)

                if angle2 is not None and std2 is not None:
                    if np.abs(np.pi - angle2) < angle2:
                        angle2 -= np.pi
                    #     self.magnified_tilt_angle -= np.pi
                    
                    angle2 *= 1e3
                    std2 *= 1e3

                    std2_temp = std2
                    c = 0
                    
                    if std != 0:
                        while int(std2_temp) < 1:
                            std2_temp *= 10
                            c += 1
                    else:
                        c = 2

                    text = f"{'Average'}: {int(angle2 * 10**c)/10**c:>10.{c}f}±{int(std2 * 10**c)/10**c:.{c}f} mrad" + additional_string
            except np.exceptions.AxisError:
                text = f"Error: not enough points entered or points/times do not generate well-defined plane"
            self.result.setText(text)
            
            pass

        def closeEvent(self, event):
            for i, row in enumerate(self.input_rows):
                values = row.get_values()
                self.settings.setValue(f"row_{row.row_id}", values)

            self.settings.setValue("velocity", self.velocity_input.value())
            self.settings.setValue("time_slider", self.slider.value())
            self.settings.setValue("color_method", self.color_method.currentIndex())
            self.settings.setValue("magnification", self.magnification.value())

            self.settings.setValue("geometry", self.saveGeometry())
            super().closeEvent(event)


    if __name__ == "__main__":
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

except Exception:
    with open("error_log.txt", "w") as f:
        f.write(traceback.format_exc())