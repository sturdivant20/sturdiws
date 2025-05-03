import sys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import make_splrep
from utils.parsers import ParseSturdrLogs, ParseNavSimStates
from utils.plotters import MyWindow, MatplotlibWidget, FoliumPlotWidget
from PyQt6 import QtWidgets
from navtools._navtools_core.attitude import quat2euler

if __name__ == "__main__":
    color_cycle = list(sns.color_palette().as_hex())
    color_cycle.append("#100c08")  # "#a2e3b8"
    # color_cycle = cycler("color", color_cycle)
    sns.set_theme(
        font="Times New Roman",
        context="paper",  # poster, talk, notebook, paper
        style="ticks",
        rc={
            "axes.grid": True,
            # "grid.linestyle": ":",
            "grid.color": "0.85",
            "lines.linewidth": 2,
            "xtick.direction": "in",
            "ytick.direction": "in",
        },
        font_scale=1.5,
        palette=sns.color_palette(color_cycle),
    )

    # parse
    nav, channels = ParseSturdrLogs("./results/USRP_LIVE", True, False)
    truth_nav = pd.read_csv(
        "./data/usrp_data/dds_record_5/ublox_pva.csv",
        skiprows=2,
        usecols=[1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    )
    truth_satnav = pd.read_csv(
        "./data/usrp_data/dds_record_5/ublox_satnav.csv", skiprows=2, usecols=[1, 7, 8]
    )

    # make truth into splines
    # fmt: off
    truth = {
        "Week": make_splrep(truth_satnav[" receiver_clock_time.seconds_of_week"], truth_satnav[" receiver_clock_time.week_number"]),
        "tR": make_splrep(truth_satnav[" receiver_clock_time.seconds_of_week"], truth_satnav[" receiver_clock_time.seconds_of_week"]),
        "satnav_t": make_splrep(truth_satnav[" receiver_clock_time.seconds_of_week"], truth_satnav[" db3_timestamp"]),
        "Lat": make_splrep(truth_nav[" db3_timestamp"], truth_nav[" p1"]),
        "Lon": make_splrep(truth_nav[" db3_timestamp"], truth_nav[" p2"]),
        "H": make_splrep(truth_nav[" db3_timestamp"], truth_nav[" p3"]),
        "vN": make_splrep(truth_nav[" db3_timestamp"], truth_nav[" v1"]),
        "vE": make_splrep(truth_nav[" db3_timestamp"], truth_nav[" v2"]),
        "vD": make_splrep(truth_nav[" db3_timestamp"], truth_nav[" v3"]),
        "qw": make_splrep(truth_nav[" db3_timestamp"], truth_nav[" quaternion[3]"]),
        "qx": make_splrep(truth_nav[" db3_timestamp"], truth_nav[" quaternion[0]"]),
        "qy": make_splrep(truth_nav[" db3_timestamp"], truth_nav[" quaternion[1]"]),
        "qz": make_splrep(truth_nav[" db3_timestamp"], truth_nav[" quaternion[2]"]),
    }
    # fmt: on

    # create window
    app = QtWidgets.QApplication(sys.argv)
    win = MyWindow("Live USRP Results")

    # open folium map
    mymap = FoliumPlotWidget(geobasemap="satellite", zoom=16)
    mymap.AddLine(
        np.array(
            [
                np.rad2deg(truth["Lat"](truth["satnav_t"](nav["tR"]))),
                np.rad2deg(truth["Lon"](truth["satnav_t"](nav["tR"]))),
            ]
        ).T.tolist(),
        color="#00FFFF",
        weight=5,
        opacity=1,
    )
    mymap.AddLine(
        np.rad2deg(nav.loc[::50][["Lat", "Lon"]].values).tolist(),
        color="#FF0000",
        weight=5,
        opacity=1,
    )
    mymap.AddLegend({"Ublox": "#00FFFF", "SturDR": "#FF0000"})
    win.NewTab(mymap, "GeoPlot")

    # Attitude
    mya = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    rpy = np.zeros((len(nav), 3), order="F")
    ubx_rpy = np.zeros((len(nav), 3), order="F")
    for ii in range(len(nav)):
        q = np.array(
            [nav.loc[ii, "qw"], nav.loc[ii, "qx"], nav.loc[ii, "qy"], nav.loc[ii, "qz"]], order="F"
        )
        rpy[ii, :] = np.rad2deg(quat2euler(q, True))
        q = np.array(
            [
                truth["qw"](truth["satnav_t"](nav.loc[ii, "tR"])),
                truth["qx"](truth["satnav_t"](nav.loc[ii, "tR"])),
                truth["qy"](truth["satnav_t"](nav.loc[ii, "tR"])),
                truth["qz"](truth["satnav_t"](nav.loc[ii, "tR"])),
            ],
            order="F",
        )
        ubx_rpy[ii, :] = np.rad2deg(quat2euler(q, True))
    sns.lineplot(x=nav["tR"], y=rpy[:, 0], label="SturDR", color="#a52a2a", ax=mya.ax[0])
    sns.lineplot(x=nav["tR"], y=ubx_rpy[:, 0], label="Ublox", color="#100c08", ax=mya.ax[0])
    mya.ax[0].set(ylabel="Roll [\N{DEGREE SIGN}]")
    mya.ax[0].minorticks_on()
    mya.ax[0].tick_params(which="minor", length=0)
    mya.ax[0].grid(which="minor", axis="both", linestyle=":", color="0.8")
    mya.ax[0].tick_params(axis="x", which="both", top=True, bottom=True)
    mya.ax[0].tick_params(axis="y", which="both", left=True, right=True)
    mya.ax[0].legend(loc="upper left")
    sns.lineplot(x=nav["tR"], y=rpy[:, 1], color="#a52a2a", ax=mya.ax[1])
    sns.lineplot(x=nav["tR"], y=ubx_rpy[:, 1], color="#100c08", ax=mya.ax[1])
    mya.ax[1].set(ylabel="Pitch [\N{DEGREE SIGN}]")
    mya.ax[1].minorticks_on()
    mya.ax[1].tick_params(which="minor", length=0)
    mya.ax[1].grid(which="minor", axis="both", linestyle=":", color="0.8")
    mya.ax[1].tick_params(axis="x", which="both", top=True, bottom=True)
    mya.ax[1].tick_params(axis="y", which="both", left=True, right=True)
    sns.lineplot(x=nav["tR"], y=rpy[:, 2], color="#a52a2a", ax=mya.ax[2])
    sns.lineplot(x=nav["tR"], y=ubx_rpy[:, 2], color="#100c08", ax=mya.ax[2])
    mya.ax[2].set(ylabel="Yaw [\N{DEGREE SIGN}]", xlabel="Time [s]")
    mya.ax[2].minorticks_on()
    mya.ax[2].tick_params(which="minor", length=0)
    mya.ax[2].grid(which="minor", axis="both", linestyle=":", color="0.8")
    mya.ax[2].tick_params(axis="x", which="both", top=True, bottom=True)
    mya.ax[2].tick_params(axis="y", which="both", left=True, right=True)
    mya.f.tight_layout()
    win.NewTab(mya, "Attitude")

    # C/No
    mycno = MatplotlibWidget(figsize=(8, 8))
    for i in range(len(channels)):
        sns.lineplot(
            x=channels[i]["t"] / 1000,
            y=channels[i]["CNo"],
            label=channels[i].loc[0, "SVID"],
            ax=mycno.ax,
        )
    mycno.ax.set(xlabel="t [s]", ylabel="C/N$_0$ [dB-Hz]")
    mycno.ax.minorticks_on()
    mycno.ax.tick_params(which="minor", length=0)
    mycno.ax.grid(which="minor", axis="both", linestyle=":", color="0.8")
    mycno.ax.tick_params(axis="x", which="both", top=True, bottom=True)
    mycno.ax.tick_params(axis="y", which="both", left=True, right=True)
    mycno.f.tight_layout()
    win.NewTab(mycno, "C/No")

    # Doppler
    mydopp = MatplotlibWidget(figsize=(8, 8))
    for i in range(len(channels)):
        sns.lineplot(
            x=channels[i]["t"] / 1000,
            y=channels[i]["Doppler"],
            label=channels[i].loc[0, "SVID"],
            ax=mydopp.ax,
        )
    mydopp.ax.set(xlabel="t [s]", ylabel="Doppler [Hz]")
    mydopp.ax.minorticks_on()
    mydopp.ax.tick_params(which="minor", length=0)
    mydopp.ax.grid(which="minor", axis="both", linestyle=":", color="0.8")
    mydopp.ax.tick_params(axis="x", which="both", top=True, bottom=True)
    mydopp.ax.tick_params(axis="y", which="both", left=True, right=True)
    mydopp.f.tight_layout()
    win.NewTab(mydopp, "Doppler")

    # Tracking Status
    mytrack = MatplotlibWidget(figsize=(8, 8))
    for i in range(len(channels)):
        sns.lineplot(
            x=channels[i]["t"] / 1000,
            y=channels[i]["TrackingStatus"],
            label=channels[i].loc[0, "SVID"],
            ax=mytrack.ax,
        )
    mytrack.ax.set(xlabel="t [s]", ylabel="Tracking Status")
    mytrack.ax.minorticks_on()
    mytrack.ax.tick_params(which="minor", length=0)
    mytrack.ax.grid(which="minor", axis="both", linestyle=":", color="0.8")
    mytrack.ax.tick_params(axis="x", which="both", top=True, bottom=True)
    mytrack.ax.tick_params(axis="y", which="both", left=True, right=True)
    mytrack.f.tight_layout()
    win.NewTab(mytrack, "Tracking Status")

    win.show()
    sys.exit(app.exec())
