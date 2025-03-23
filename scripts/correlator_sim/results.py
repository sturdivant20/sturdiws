import sys
from parsers import ParseCorrelatorSimLogs, ParseNavStates
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PyQt6 import QtWidgets
from plotters import FoliumPlotWidget, MatplotlibWidget, SkyPlot


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


if __name__ == "__main__":
    # COLORS = ["#100c08", "#a52a2a", "#a2e3b8"]
    sns.set_theme(
        font="Times New Roman",
        context="paper",
        style="ticks",
        rc={
            "axes.grid": True,
            "grid.linestyle": ":",
            "lines.linewidth": 2,
        },
        font_scale=1.5,
    )

    # i know these are the satellites in the 'sim_ephem.bin' file
    svid = ["GPS1", "GPS17", "GPS30", "GPS14", "GPS7", "GPS21", "GPS19", "GPS13"]

    # parse results
    nav, err, var, channels = ParseCorrelatorSimLogs("results/VT_ARRAY_CORRELATOR_SIM/1/", True)
    truth = ParseNavStates("data/sim_truth.bin")

    # create window
    app = QtWidgets.QApplication(sys.argv)
    win = MyWindow()

    # open folium map
    nav_data = nav.loc[::50][["lat", "lon"]].values.tolist()
    mymap = FoliumPlotWidget(geobasemap="satellite", zoom=16)
    mymap.AddLine(
        truth.loc[::100][["lat", "lon"]].values.tolist(), color="#00FFFF", weight=5, opacity=1
    )
    mymap.AddLine(
        nav.loc[::50][["lat", "lon"]].values.tolist(), color="#FF0000", weight=5, opacity=1
    )
    mymap.AddLegend({"Truth": "#00FFFF", "Estimate": "#FF0000"})
    win.NewTab(mymap, "GeoPlot")

    # plot channel data
    myv = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=truth["t"], y=truth["vn"], label="Truth", color="#100c08", ax=myv.ax[0])
    sns.lineplot(x=nav["t"], y=nav["vn"], label="EKF", color="#a52a2a", ax=myv.ax[0])
    myv.ax[0].set(ylabel="North [m/s]")
    sns.lineplot(x=truth["t"], y=truth["ve"], color="#100c08", ax=myv.ax[1])
    sns.lineplot(x=nav["t"], y=nav["ve"], color="#a52a2a", ax=myv.ax[1])
    myv.ax[1].set(ylabel="East [m/s]")
    sns.lineplot(x=truth["t"], y=truth["vd"], color="#100c08", ax=myv.ax[2])
    sns.lineplot(x=nav["t"], y=nav["vd"], color="#a52a2a", ax=myv.ax[2])
    myv.ax[2].set(ylabel="Down [m/s]", xlabel="Time [s]")
    myv.f.tight_layout()
    win.NewTab(myv, "Velocity")

    mya = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=truth["t"], y=truth["r"], label="Truth", color="#100c08", ax=mya.ax[0])
    sns.lineplot(x=nav["t"], y=nav["r"], label="EKF", color="#a52a2a", ax=mya.ax[0])
    mya.ax[0].set(ylabel="Roll [deg]")
    sns.lineplot(x=truth["t"], y=truth["p"], color="#100c08", ax=mya.ax[1])
    sns.lineplot(x=nav["t"], y=nav["p"], color="#a52a2a", ax=mya.ax[1])
    mya.ax[1].set(ylabel="Pitch [deg]")
    sns.lineplot(x=truth["t"], y=truth["y"], color="#100c08", ax=mya.ax[2])
    sns.lineplot(x=nav["t"], y=nav["y"], color="#a52a2a", ax=mya.ax[2])
    mya.ax[2].set(ylabel="Yaw [deg]", xlabel="Time [s]")
    mya.f.tight_layout()
    win.NewTab(mya, "Attitude")

    myperr = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=err["t"], y=err["lat"], label="$\\mu$", color="#100c08", ax=myperr.ax[0])
    sns.lineplot(
        x=var["t"], y=3 * np.sqrt(var["lat"]), label="$3\\sigma$", color="#a52a2a", ax=myperr.ax[0]
    )
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["lat"]), color="#a52a2a", ax=myperr.ax[0])
    myperr.ax[0].set(ylabel="North [m]")  # ylim=[-3, 3],
    sns.lineplot(x=err["t"], y=err["lon"], color="#100c08", ax=myperr.ax[1])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["lon"]), color="#a52a2a", ax=myperr.ax[1])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["lon"]), color="#a52a2a", ax=myperr.ax[1])
    myperr.ax[1].set(ylabel="East [m]")  # ylim=[-3, 3],
    sns.lineplot(x=err["t"], y=err["h"], color="#100c08", ax=myperr.ax[2])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["h"]), color="#a52a2a", ax=myperr.ax[2])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["h"]), color="#a52a2a", ax=myperr.ax[2])
    myperr.ax[2].set(ylabel="Down [m]", xlabel="Time [s]")  # ylim=[-6, 6],
    myperr.f.tight_layout()
    win.NewTab(myperr, "Position Error")

    myverr = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=err["t"], y=err["vn"], label="$\\mu$", color="#100c08", ax=myverr.ax[0])
    sns.lineplot(
        x=var["t"], y=3 * np.sqrt(var["vn"]), label="$3\\sigma$", color="#a52a2a", ax=myverr.ax[0]
    )
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["vn"]), color="#a52a2a", ax=myverr.ax[0])
    myverr.ax[0].set(ylabel="North [m/s]")  # ylim=[-3, 3],
    sns.lineplot(x=err["t"], y=err["ve"], color="#100c08", ax=myverr.ax[1])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["ve"]), color="#a52a2a", ax=myverr.ax[1])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["ve"]), color="#a52a2a", ax=myverr.ax[1])
    myverr.ax[1].set(ylabel="East [m/s]")  # ylim=[-3, 3],
    sns.lineplot(x=err["t"], y=err["vd"], color="#100c08", ax=myverr.ax[2])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["vd"]), color="#a52a2a", ax=myverr.ax[2])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["vd"]), color="#a52a2a", ax=myverr.ax[2])
    myverr.ax[2].set(ylabel="Down [m/s]", xlabel="Time [s]")  # ylim=[-6, 6],
    myverr.f.tight_layout()
    win.NewTab(myverr, "Velocity Error")

    myaerr = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    sns.lineplot(x=err["t"], y=err["r"], label="$\\mu$", color="#100c08", ax=myaerr.ax[0])
    sns.lineplot(
        x=var["t"], y=3 * np.sqrt(var["r"]), label="$3\\sigma$", color="#a52a2a", ax=myaerr.ax[0]
    )
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["r"]), color="#a52a2a", ax=myaerr.ax[0])
    myaerr.ax[0].set(ylabel="Roll [deg]")  # ylim=[-3, 3],
    sns.lineplot(x=err["t"], y=err["p"], color="#100c08", ax=myaerr.ax[1])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["p"]), color="#a52a2a", ax=myaerr.ax[1])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["p"]), color="#a52a2a", ax=myaerr.ax[1])
    myaerr.ax[1].set(ylabel="Pitch [deg]")  # ylim=[-3, 3],
    sns.lineplot(x=err["t"], y=err["y"], color="#100c08", ax=myaerr.ax[2])
    sns.lineplot(x=var["t"], y=3 * np.sqrt(var["y"]), color="#a52a2a", ax=myaerr.ax[2])
    sns.lineplot(x=var["t"], y=-3 * np.sqrt(var["y"]), color="#a52a2a", ax=myaerr.ax[2])
    myaerr.ax[2].set(ylabel="Yaw [deg]", xlabel="Time [s]")  # ylim=[-6, 6],
    myaerr.f.tight_layout()
    win.NewTab(myaerr, "Attitude Error")

    mycno = MatplotlibWidget(figsize=(8, 8))
    for i in range(len(channels)):
        sns.lineplot(x=channels[i]["t"], y=channels[i]["est_cno"], label=svid[i], ax=mycno.ax)
    # sns.lineplot(x=channels[0]["t"], y=channels[0]["true_cno"], label="Truth", ax=mycno.ax)
    # sns.lineplot(x=channels[0]["t"], y=channels[0]["est_cno"], label="Estimated", ax=mycno.ax)
    # sns.lineplot(x=channels[0]["t"], y=channels[0]["est_cno_bs"], label="Beam Steered", ax=ax0)
    mycno.ax.set(xlabel="t [s]", ylabel="C/N$_0$ [dB-Hz]")
    mycno.f.tight_layout()
    win.NewTab(mycno, "C/No")

    mycorr = MatplotlibWidget(figsize=(8, 8))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i in range(len(channels)):
        sns.lineplot(
            x=channels[i]["t"],
            y=channels[i]["IP"] ** 2 + channels[i]["QP"] ** 2,
            label=svid[i],
            color=color_cycle[i],
            ax=mycorr.ax,
        )
        sns.lineplot(
            x=channels[i]["t"],
            y=channels[i]["IP_reg_0"] ** 2 + channels[i]["QP_reg_0"] ** 2,
            label=None,
            color=color_cycle[i],
            ax=mycorr.ax,
        )
    mycorr.ax.set(xlabel="t [s]", ylabel="Correlator Power")
    mycorr.f.tight_layout()
    win.NewTab(mycorr, "Correlator Power")

    mypolar = MatplotlibWidget(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    for i in range(len(channels)):
        mypolar.ax = SkyPlot(
            channels[i]["az"].values, channels[i]["el"].values, [svid[i]], ax=mypolar.ax
        )
    win.NewTab(mypolar, "SkyPlot")

    # open plots
    # plt.show()
    win.show()
    sys.exit(app.exec())
