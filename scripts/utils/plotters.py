import sys
import io
import json
from statistics import mean, median
import folium
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtWebEngineWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
from collections import deque
from .parsers import ParseNavSimStates

# Good folium map (tile) options
# "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/tile/{z}/{y}/{x}"
# "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
# "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My Window")
        self.tab_widget = QtWidgets.QTabWidget()
        self.tabs = []
        self.i = 0
        self.setCentralWidget(self.tab_widget)

    def NewTab(self, widget: QtWidgets.QWidget, tab_name: str = None):
        if type(widget) == MatplotlibWidget:
            layout = QtWidgets.QVBoxLayout()
            toolbar = NavigationToolbar2QT(widget, self)
            layout.addWidget(toolbar)
            layout.addWidget(widget)
            w = QtWidgets.QWidget()
            w.setLayout(layout)
            self.tabs.append(w)
        else:
            self.tabs.append(widget)
        if tab_name is None:
            tab_name = f"Tab {self.i}"
        self.tab_widget.addTab(self.tabs[-1], tab_name)
        self.i += 1


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
        zoom : zoom level of the map
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
        if "zoom" not in kwargs:
            kwargs["zoom"] = 15
        self.kwargs = kwargs

        # svg of an arrow
        self.svg = f"""<svg version="1.1" id="svg1" width="22" height="26.4" viewBox="0 0 131.51233 157.83148" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg"><defs id="defs1" /><g id="g1" transform="translate(-1.3988492,-1.2305383)"><path style="fill:{self.kwargs['color']}" d="M 1.4932329,158.47237 C 3.9473229,151.73955 67.012636,1.2305453 67.379686,1.2305383 68.068156,1.2305253 133.39492,158.532 132.90847,159.01846 c -0.21492,0.21491 -15.0473,-8.4092 -32.960854,-19.1647 -32.5701,-19.55545 -32.5701,-19.55545 -64.319024,-0.47891 -33.1717391,19.93146 -34.76403912,20.8223 -34.1353591,19.09752 z" id="path1" /></g></svg>"""

        self.BuildMap()

    def BuildMap(self):
        # create the folium map
        x, y = zip(*self.coord)
        l = len(x)
        loc = [sum(x) / l, sum(y) / l]
        if self.kwargs["geobasemap"].casefold() == "streets":
            self.map = folium.Map(
                title="Folium Map",
                location=loc,
                zoom_start=self.kwargs["zoom"],
            )
        else:
            self.map = folium.Map(
                title="Folium Map",
                location=loc,
                zoom_start=self.kwargs["zoom"],
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
                    iconSize: [22, 26.4],
                    iconAnchor: [11, 13.2]
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
                        iconSize: [22, 26.4],
                        iconAnchor: [11, 13.2]
                    }}); 
                    window.{self.arrow_marker_name}.setIcon(customIcon); 
                }}
            """
            self.page().runJavaScript(js_code)


class FoliumPlotWidget(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, **kwargs):
        super().__init__()

        # init map
        if "geobasemap" not in kwargs:
            kwargs["geobasemap"] = "streets"
        if "zoom" not in kwargs:
            kwargs["zoom"] = 15
        if kwargs["geobasemap"].casefold() == "streets":
            self.map = folium.Map(
                title="Folium Map",
                location=[0, 0],
                zoom_start=kwargs["zoom"],
            )
        else:
            self.map = folium.Map(
                title="Folium Map",
                location=[0, 0],
                zoom_start=kwargs["zoom"],
                tiles="https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="World Imagery",
            )

        data = io.BytesIO()
        self.map.save(data, close_file=False)
        self.setHtml(data.getvalue().decode())

    def AddLine(self, coord: list[tuple], **kwargs):
        # if self.map.location == [0, 0]:
        x, y = zip(*coord)
        x_max = max(x)
        x_min = min(x)
        y_max = max(y)
        y_min = min(y)

        loc = [(x_max + x_min) / 2, (y_max + y_min) / 2]
        self.map.location = loc
        self.map.fit_bounds([[x_min, y_min], [x_max, y_max]])

        if "color" not in kwargs:
            kwargs["color"] = "#000000"
        folium.PolyLine(locations=coord, **kwargs).add_to(self.map)

        data = io.BytesIO()
        self.map.save(data, close_file=False)
        self.setHtml(data.getvalue().decode())
        return

    def AddLegend(self, ldict: dict):
        legend_html = f"""
            <div style="position: fixed; 
                top: 0px; right: 0px; width: 140px; height: 60px; 
                border:2px solid grey; z-index:9999; font-size:20px;
                background-color:white; opacity: 0.85;">
            """
        for key, val in ldict.items():
            legend_html += f"""
                &nbsp; {key} &nbsp; <i class="fa fa-circle" style="color:{val}"></i><br>
                """
        legend_html += "</div>"
        self.map.get_root().html.add_child(folium.Element(legend_html))
        data = io.BytesIO()
        self.map.save(data, close_file=False)
        self.setHtml(data.getvalue().decode())
        return


class MatplotlibWidget(FigureCanvasQTAgg):
    def __init__(self, **kwargs):
        self.f, self.ax = plt.subplots(**kwargs)
        super().__init__(self.f)


def SkyPlot(
    az: np.ndarray | list,
    el: np.ndarray | list,
    labels: np.ndarray | list = None,
    ax: plt.Axes = None,
    **kwargs,
):
    """
    Plots satellite positions on polar plot given azimuth and elevation from receiver to satellites
    """

    if isinstance(az, list):
        az = np.asarray(az)
    if isinstance(el, list):
        el = np.asarray(el)
    az = np.deg2rad(np.atleast_2d(az)).astype(np.double)
    el = np.atleast_2d(el).astype(np.double)
    if az.shape[0] == 1:
        az = az.T
    if el.shape[0] == 1:
        el = el.T

    # make sure axes exists
    if ax == None or ax.name != "polar":
        ax = plt.subplot(projection="polar")

    # format polar axes
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(90, 0)
    r_labels = [
        "0\N{DEGREE SIGN}",
        "20\N{DEGREE SIGN}",
        "40\N{DEGREE SIGN}",
        "60\N{DEGREE SIGN}",
        "80\N{DEGREE SIGN}",
    ]
    ax.set_rgrids(range(0, 91, 20), r_labels, angle=-90)
    ax.set_xticks(np.deg2rad(np.linspace(180, -180, 12, endpoint=False)))
    ax.set_thetalim(-np.pi, np.pi)
    ax.set_xticklabels(
        [
            "S",
            "150\N{DEGREE SIGN}",
            "120\N{DEGREE SIGN}",
            "E",
            "60\N{DEGREE SIGN}",
            "30\N{DEGREE SIGN}",
            "N",
            "330\N{DEGREE SIGN}",
            "300\N{DEGREE SIGN}",
            "W",
            "240\N{DEGREE SIGN}",
            "210\N{DEGREE SIGN}",
        ]
    )
    ax.set_axisbelow(True)

    # check kwargs
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.6
    if "s" not in kwargs:
        kwargs["s"] = 100

    # plot the data
    for i in range(az.shape[1]):
        (line,) = ax.plot(az[:, i], el[:, i])
        kwargs["color"] = line.get_color()
        ax.scatter(az[-1, i], el[-1, i], **kwargs)
        t = ax.annotate(
            labels[i],
            xy=[
                az[-1, i],
                el[-1, i],
            ],
            fontsize=10,
            ha="center",
            va="bottom",
            color=kwargs["color"],
        )
        p = t.get_position()
        p_ax = ax.transData.transform(p)
        p_ax_new = p_ax + np.array([0, kwargs["s"] / 25])
        p_new = ax.transData.inverted().transform(p_ax_new) + np.array([0, 180])
        t.set_position(p_new)
    return ax


if __name__ == "__main__":

    class MyWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("folium example")

            # load gps data
            self.gps = ParseNavSimStates("data/sim_truth.bin")[4000:-4000]
            kwargs = {
                "color": "#ff0000",
                "weight": 5,
                "opacity": 1,
                "geobasemap": "satellite",
                "zoom": 20,
            }
            self.i = 4000

            # create tabs
            self.tab_widget = QtWidgets.QTabWidget()
            self.tabs = [
                FoliumWidget([self.gps.loc[self.i, ["lat", "lon"]].values.tolist()], **kwargs),
                pg.GraphicsLayoutWidget(),
            ]

            # create 3x1 velocity plots
            styles = {"color": "#00ff00", "font-size": "18px"}
            mypen = pg.mkPen("#ff0000", width=3, style=QtCore.Qt.PenStyle.DashLine)
            self.time = deque(np.arange(-30, 1).tolist(), maxlen=31)
            self.vel = [
                deque([self.gps["vn"][self.i]] * 31, maxlen=31),
                deque([self.gps["ve"][self.i]] * 31, maxlen=31),
                deque([self.gps["vd"][self.i]] * 31, maxlen=31),
            ]
            self.vel_ax = [
                self.tabs[1].addPlot(row=0, col=0),
                self.tabs[1].addPlot(row=1, col=0),
                self.tabs[1].addPlot(row=2, col=0),
            ]
            self.vel_plots = [
                self.vel_ax[0].plot(
                    self.time,
                    self.vel[0],
                    name=f"vN",
                    pen=mypen,
                    symbolBrush="#ff0000",
                    symbolPen="#ffffff",
                    symbol="t2",
                    symbolSize=10,
                ),
                self.vel_ax[1].plot(
                    self.time,
                    self.vel[1],
                    name=f"vE",
                    pen=mypen,
                    symbolBrush="#ff0000",
                    symbolPen="#ffffff",
                    symbol="t2",
                    symbolSize=10,
                ),
                self.vel_ax[2].plot(
                    self.time,
                    self.vel[2],
                    name=f"vD",
                    pen=mypen,
                    symbolBrush="#ff0000",
                    symbolPen="#ffffff",
                    symbol="t2",
                    symbolSize=10,
                ),
            ]
            self.vel_ax[0].setLabel("left", "vN [m/s]", **styles)
            self.vel_ax[0].showGrid(x=True, y=True)
            self.vel_ax[0].setXRange(-30, 0)
            self.vel_ax[0].getAxis("bottom").setStyle(showValues=False)
            self.vel_ax[0].enableAutoRange(axis="y")
            self.vel_ax[0].setAutoVisible(y=True)
            self.vel_ax[1].setLabel("left", "vE [m/s]", **styles)
            self.vel_ax[1].showGrid(x=True, y=True)
            self.vel_ax[1].setXRange(-30, 0)
            self.vel_ax[1].getAxis("bottom").setStyle(showValues=False)
            self.vel_ax[1].enableAutoRange(axis="y")
            self.vel_ax[1].setAutoVisible(y=True)
            self.vel_ax[2].setLabel("left", "vD [m/s]", **styles)
            self.vel_ax[2].setLabel("bottom", "Time [s]", **styles)
            self.vel_ax[2].showGrid(x=True, y=True)
            self.vel_ax[2].setXRange(-30, 0)
            self.vel_ax[2].enableAutoRange(axis="y")
            self.vel_ax[2].setAutoVisible(y=True)
            self.vel_ax[0].setXLink(self.vel_ax[1])
            self.vel_ax[0].setXLink(self.vel_ax[2])
            self.vel_ax[1].setXLink(self.vel_ax[2])
            # self.vel_ax[0].setYLink(self.vel_ax[1])
            # self.vel_ax[0].setYLink(self.vel_ax[2])
            # self.vel_ax[1].setYLink(self.vel_ax[2])
            self.tabs[1].ci.layout.setRowStretchFactor(0, 6)
            self.tabs[1].ci.layout.setRowStretchFactor(1, 6)
            self.tabs[1].ci.layout.setRowStretchFactor(2, 8)

            # add tabs to the tab widget
            self.tab_widget.addTab(self.tabs[0], f"Map")
            self.tab_widget.addTab(self.tabs[1], f"Velocity")
            self.setCentralWidget(self.tab_widget)
            self.i += 100

            # simulate new measurements
            self.timer = QtCore.QTimer()
            self.timer.setInterval(100)
            self.timer.timeout.connect(self.update)
            self.timer.start()

        def update(self):
            # update map
            tmp = self.gps.loc[self.i, ["lat", "lon"]].values.tolist()
            heading = np.mod(self.gps.loc[self.i, "y"], 360)
            if not isinstance(tmp[0], list):
                tmp = [tmp]
            self.i += 100
            self.tabs[0].UpdateMap(tmp, heading)

            # update velocity
            self.vel[0].append(self.gps["vn"][self.i])
            self.vel[1].append(self.gps["ve"][self.i])
            self.vel[2].append(self.gps["vd"][self.i])
            self.vel_plots[0].setData(self.time, self.vel[0])
            self.vel_plots[1].setData(self.time, self.vel[1])
            self.vel_plots[2].setData(self.time, self.vel[2])

    app = QtWidgets.QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec())
