import sys
import io
import json
import folium
import numpy as np
from parsers import ParseNavStates
from PyQt6 import QtWidgets, QtWebEngineWidgets, QtCore

# Good folium map (tile) options
# "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/tile/{z}/{y}/{x}"
# "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
# "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


class FoliumWidget(QtWebEngineWidgets.QWebEngineView):
    """
    FoliumWidget
    ============

    Custom folium wrapper that allows a custom map and trajectory to be dynamically updated within
    a PyQt6 GUI.

    `kwargs` includes but is not limited to:
        geobasemap : map tiles ("streets" or "satellite")
        color : color of trajectory line, default is "#ff0000"
        weight : weight of the trajectory line
        opacity : transparency of the trajectory line
    """

    coord: list = []
    map: folium.Map
    layer_name: str
    arrow_marker_name: str
    svg: str
    is_map_loaded: False
    kwargs: dict

    def __init__(self, coord: list[tuple] = [(0, 0)], **kwargs):
        super().__init__()
        self.coord = coord
        self.is_map_loaded = False
        self.layer_name = "custom_polyline"
        self.arrow_marker_name = "arrow_marker"

        # make sure a color and geobasemap have been defined
        if "color" not in kwargs:
            kwargs["color"] = "#ff000"
        if "geobasemap" not in kwargs:
            kwargs["geobasemap"] = "streets"
        self.kwargs = kwargs

        # svg of an arrow
        self.svg = f"""<svg version="1.1" id="svg1" width="20" height="24" viewBox="0 0 131.51233 157.83148" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg"><defs id="defs1" /><g id="g1" transform="translate(-1.3988492,-1.2305383)"><path style="fill:{self.kwargs['color']}" d="M 1.4932329,158.47237 C 3.9473229,151.73955 67.012636,1.2305453 67.379686,1.2305383 68.068156,1.2305253 133.39492,158.532 132.90847,159.01846 c -0.21492,0.21491 -15.0473,-8.4092 -32.960854,-19.1647 -32.5701,-19.55545 -32.5701,-19.55545 -64.319024,-0.47891 -33.1717391,19.93146 -34.76403912,20.8223 -34.1353591,19.09752 z" id="path1" /></g></svg>"""

        self.BuildMap()

    def BuildMap(self):
        # create the folium map
        if self.kwargs["geobasemap"].casefold() == "streets":
            self.map = folium.Map(
                title="Folium Map",
                location=self.coord[-1],
                zoom_start=20,
            )
        else:
            self.map = folium.Map(
                title="Folium Map",
                location=self.coord[-1],
                zoom_start=20,
                tiles="https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="World Imagery",
            )

        # turn folium map into html and display
        data = io.BytesIO()
        self.map.save(data, close_file=False)
        self.setHtml(data.getvalue().decode())
        self.page().loadFinished.connect(self.IsMapLoaded)

    def IsMapLoaded(self):
        # once basemap has been loaded, begin adding a trajectory line on top
        self.is_map_loaded = True
        js_code = f"""
            function createPolyline() {{
                window.{self.layer_name} = L.polyline({json.dumps(self.coord)}, {self.kwargs}).addTo({self.map.get_name()});
                window.polyline_initialized = true;
                var customIcon = L.divIcon({{
                    html: '<div style="transform: rotate(0deg);">{self.svg}</div>',
                    className: 'dummy',
                    iconSize: [20, 24],
                    iconAnchor: [10, 12]
                }});
                window.{self.arrow_marker_name} = L.marker({json.dumps(self.coord[-1])}, {{icon: customIcon}}).addTo({self.map.get_name()});
            }}
            createPolyline();
        """
        self.page().runJavaScript(js_code)

    def UpdateMap(self, coord: list[tuple], heading: float):
        # only save the last 30 points
        self.coord.extend(coord)
        if len(self.coord) > 30:
            self.coord = self.coord[-30:]

        # move the map and update the line
        if self.is_map_loaded:
            self.map.location = self.coord[-1]

            js_coords = json.dumps([list(coord) for coord in self.coord])
            js_code = f"""
                {self.map.get_name()}.setView({json.dumps(self.map.location)}, {self.map.options['zoom']});
                if (window.{self.layer_name}) {{ 
                    window.{self.layer_name}.setLatLngs({js_coords}); 
                }}
                if (window.{self.arrow_marker_name}) {{ 
                    window.{self.arrow_marker_name}.setLatLng({json.dumps(self.coord[-1])}); 
                    var customIcon = L.divIcon({{
                        html: '<div style="transform: rotate(' + {heading} + 'deg);">{self.svg}</div>',
                        className: 'dummy',
                        iconSize: [20, 24],
                        iconAnchor: [10, 12]
                    }}); 
                    window.{self.arrow_marker_name}.setIcon(customIcon); 
                }}
            """
            self.page().runJavaScript(js_code)


if __name__ == "__main__":

    class MyWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("folium example")

            # load gps data
            self.gps = ParseNavStates("data/sim_truth.bin")
            kwargs = {"color": "#ff0000", "weight": 5, "opacity": 1, "geobasemap": "satellite"}

            # create folium map
            self.i = 0
            self.map_widget = FoliumWidget(
                [self.gps.loc[self.i, ["lat", "lon"]].values.tolist()], **kwargs
            )
            self.setCentralWidget(self.map_widget)
            self.i += 100

            # simulate new measurements
            self.timer = QtCore.QTimer()
            self.timer.setInterval(200)
            self.timer.timeout.connect(self.update)
            self.timer.start()

        def update(self):
            tmp = self.gps.loc[self.i, ["lat", "lon"]].values.tolist()
            heading = np.mod(self.gps.loc[self.i, "y"], 360)
            print(f"{heading}")
            if not isinstance(tmp[0], list):
                tmp = [tmp]
            self.i += 100
            self.map_widget.UpdateMap(tmp, heading)

    app = QtWidgets.QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec())


# import pyqtgraph as pg
# from PyQt6 import QtWidgets, QtCore
# import numpy as np

# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("tabs example")

#         # data
#         self.time = np.arange(31).tolist()
#         self.temperature = np.zeros((2, 31)).tolist()
#         styles = {"color": "#00ff00", "font-size": "18px"}
#         pens = [
#             pg.mkPen("#ff0000", width=3, style=QtCore.Qt.PenStyle.DashLine),
#             pg.mkPen("#0000ff", width=3, style=QtCore.Qt.PenStyle.DashLine),
#         ]

#         # tab widgets
#         self.tab_widget = QtWidgets.QTabWidget()
#         self.layouts = []
#         self.plotitems = []
#         self.plots = []
#         for i in range(2):

#             # custom tab layouts/plots
#             self.layouts.append(pg.GraphicsLayoutWidget())
#             # self.layouts[i].setBackground("w")
#             self.plotitems.append(
#                 [
#                     self.layouts[i].addPlot(row=0, col=0, pen=pens[i]),
#                     self.layouts[i].addPlot(row=1, col=0, pen=pens[i]),
#                 ]
#             )
#             self.plots.append(
#                 [
#                     self.plotitems[i][0].plot(
#                         self.time,
#                         self.temperature[i][:],
#                         name=f"Plot{i}0",
#                         pen=pens[i],
#                     ),
#                     self.plotitems[i][1].plot(
#                         self.time,
#                         self.temperature[i][:],
#                         name=f"Plot{i}1",
#                         pen=pens[i],
#                     ),
#                 ]
#             )
#             self.plotitems[i][0].setLabel("left", "Temperature (°C)", **styles)
#             self.plotitems[i][0].showGrid(x=True, y=True)
#             self.plotitems[i][0].setXRange(1, 30)
#             self.plotitems[i][0].setYRange(20, 40)
#             self.plotitems[i][1].setLabel("left", "Temperature (°C)", **styles)
#             self.plotitems[i][1].setLabel("bottom", "Time (min)", **styles)
#             self.plotitems[i][1].showGrid(x=True, y=True)
#             self.plotitems[i][1].setXRange(1, 30)
#             self.plotitems[i][1].setYRange(20, 40)
#             self.plotitems[i][0].setXLink(self.plotitems[i][1])
#             self.plotitems[i][0].setYLink(self.plotitems[i][1])

#             self.plotitems[i][0].getAxis("bottom").setStyle(showValues=False)
#             self.layouts[i].ci.layout.setRowStretchFactor(0, 5)
#             self.layouts[i].ci.layout.setRowStretchFactor(1, 7)

#             # self.tabs[i].setLayout(self.layouts[i])
#             self.tab_widget.addTab(self.layouts[i], f"Tab {i}")

#         self.setCentralWidget(self.tab_widget)

#         # simulate new measurements
#         self.timer = QtCore.QTimer()
#         self.timer.setInterval(500)
#         self.timer.timeout.connect(self.update_plots)
#         self.timer.start()

#     def update_plots(self):
#         # self.time = self.time[1:]
#         # self.time.append(self.time[-1] + 1)

#         for i in range(2):
#             self.temperature[i][:] = self.temperature[i][1:]
#             self.temperature[i].append(np.random.randint(20, 40))
#             self.plots[i][0].setData(self.time, self.temperature[i][:])
#             self.plots[i][1].setData(self.time, self.temperature[i][:])

# app = QtWidgets.QApplication([])
# main = MainWindow()
# main.show()
# app.exec()
