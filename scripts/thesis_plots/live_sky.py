"""
Show that my algorithms work on a live-sky dataset
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import make_splrep
from PyQt6 import QtWidgets
from navtools._navtools_core.attitude import quat2euler
from navtools._navtools_core.frames import lla2ned
from navtools._navtools_core.math import quatdot, quatinv
from navtools import RAD2DEG
from pathlib import Path

sys.path.append("scripts")
from utils.parsers import ParseSturdrLogs, ParseNavSimStates
from utils.plotters import MyWindow, MatplotlibWidget, FoliumPlotWidget

if __name__ == "__main__":
    SAVE = False

    color_cycle = list(sns.color_palette().as_hex())
    color_cycle.append("#100c08")  # "#a2e3b8"
    # color_cycle = cycler("color", color_cycle)
    sns.set_theme(
        font="Times New Roman",
        context="paper",  # poster, talk, notebook, paper
        style="ticks",
        rc={
            "axes.grid": True,
            "grid.linestyle": ":",
            "lines.linewidth": 2.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.default": "it",
        },
        font_scale=2.0,
        palette=sns.color_palette(color_cycle),
    )

    # parse
    nav, channels = ParseSturdrLogs("./results/USRP_LIVE", True, False)
    truth_nav = pd.read_csv(
        "/media/daniel/Sturdivant/Signal-Data/feb7CrpaCollect/dds_record_5/ublox_pva.csv",
        skiprows=2,
        usecols=[1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    )
    truth_satnav = pd.read_csv(
        "/media/daniel/Sturdivant/Signal-Data/feb7CrpaCollect/dds_record_5/ublox_satnav.csv",
        skiprows=2,
        usecols=[1, 7, 8],
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

    # calculations
    rpy = np.zeros((len(nav), 3), order="F")
    ubx_rpy = np.zeros((len(nav), 3), order="F")
    rpy_err = np.zeros((len(nav), 3), order="F")
    ned_err = np.zeros((len(nav), 3), order="F")
    nedv_err = np.zeros((len(nav), 3), order="F")
    for ii in range(len(nav)):
        q = np.array(
            [nav.loc[ii, "qw"], nav.loc[ii, "qx"], nav.loc[ii, "qy"], nav.loc[ii, "qz"]], order="F"
        )
        rpy[ii, :] = np.rad2deg(quat2euler(q, True))
        q_ubx = np.array(
            [
                truth["qw"](truth["satnav_t"](nav.loc[ii, "tR"])),
                truth["qx"](truth["satnav_t"](nav.loc[ii, "tR"])),
                truth["qy"](truth["satnav_t"](nav.loc[ii, "tR"])),
                truth["qz"](truth["satnav_t"](nav.loc[ii, "tR"])),
            ],
            order="F",
        )
        ubx_rpy[ii, :] = np.rad2deg(quat2euler(q_ubx, True))
        rpy_err[ii, :] = quat2euler(quatdot(q_ubx, quatinv(q)), True) * RAD2DEG
        lla = np.array([nav.loc[ii, "Lat"], nav.loc[ii, "Lon"], nav.loc[ii, "H"]], order="F")
        lla_ubx = np.array(
            [
                truth["Lat"](truth["satnav_t"](nav.loc[ii, "tR"])),
                truth["Lon"](truth["satnav_t"](nav.loc[ii, "tR"])),
                truth["H"](truth["satnav_t"](nav.loc[ii, "tR"])),
            ],
            order="F",
        )
        ned_err[ii, :] = lla2ned(lla, lla_ubx)
        nedv_err[ii, :] = [
            truth["vN"](truth["satnav_t"](nav.loc[ii, "tR"])) - nav.loc[ii, "vN"],
            truth["vE"](truth["satnav_t"](nav.loc[ii, "tR"])) - nav.loc[ii, "vE"],
            truth["vD"](truth["satnav_t"](nav.loc[ii, "tR"])) - nav.loc[ii, "vD"],
        ]
    course = np.rad2deg(np.atan2(nav.loc[:, "vE"], nav.loc[:, "vN"]))
    course_invalid = np.sqrt(nav.loc[:, "vE"] ** 2 + nav.loc[:, "vN"] ** 2) < 1.5
    course[course_invalid] = np.nan

    # print error stats
    ned_err[:, 2] += 20.0
    pos_err_norm = np.linalg.norm(ned_err, axis=1)
    vel_err_norm = np.linalg.norm(nedv_err, axis=1)
    print("Position:")
    print(f"RMSE: {np.sqrt(np.mean(pos_err_norm**2))}")
    print(f"Stdev.: {np.std(pos_err_norm)}")
    print(f"Max.: {np.max(np.abs(pos_err_norm))}")
    print("Velocity:")
    print(f"RMSE: {np.sqrt(np.mean(vel_err_norm**2))}")
    print(f"Stdev.: {np.std(vel_err_norm)}")
    print(f"Max.: {np.max(np.abs(vel_err_norm))}")
    print("Yaw:")
    print(f"RMSE: {np.sqrt(np.mean(rpy_err[:,2]**2))}")
    print(f"Stdev.: {np.std(rpy_err[:,2])}")
    print(f"Max.: {np.max(np.abs(rpy_err[:,2]))}")

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
    mya = MatplotlibWidget(nrows=3, ncols=1, figsize=(10, 6), sharex=True)
    sns.lineplot(x=nav["tR"], y=rpy[:, 0], color="#a52a2a", ax=mya.ax[0])
    sns.lineplot(x=nav["tR"], y=ubx_rpy[:, 0], color="#100c08", ax=mya.ax[0])
    mya.ax[0].set(ylabel="Roll [\N{DEGREE SIGN}]")
    mya.ax[0].minorticks_on()
    mya.ax[0].tick_params(which="minor", length=0)
    mya.ax[0].grid(which="minor", axis="both", linestyle=":", color="0.8")
    mya.ax[0].tick_params(axis="x", which="both", top=True, bottom=True)
    mya.ax[0].tick_params(axis="y", which="both", left=True, right=True)
    sns.lineplot(x=nav["tR"], y=rpy[:, 1], color="#a52a2a", ax=mya.ax[1])
    sns.lineplot(x=nav["tR"], y=ubx_rpy[:, 1], color="#100c08", ax=mya.ax[1])
    mya.ax[1].set(ylabel="Pitch [\N{DEGREE SIGN}]")
    mya.ax[1].minorticks_on()
    mya.ax[1].tick_params(which="minor", length=0)
    mya.ax[1].grid(which="minor", axis="both", linestyle=":", color="0.8")
    mya.ax[1].tick_params(axis="x", which="both", top=True, bottom=True)
    mya.ax[1].tick_params(axis="y", which="both", left=True, right=True)
    sns.lineplot(x=nav["tR"], y=course, color="#a2e3b8", ax=mya.ax[2], label="Course")
    sns.lineplot(x=nav["tR"], y=rpy[:, 2], color="#a52a2a", ax=mya.ax[2], label="SturDR")
    sns.lineplot(x=nav["tR"], y=ubx_rpy[:, 2], color="#100c08", ax=mya.ax[2], label="Ublox")
    # mya.ax[2].legend(loc="upper left")
    sns.move_legend(mya.ax[2], "upper center", bbox_to_anchor=(0.5, 3.7), ncol=3)
    mya.ax[2].set(ylabel="Yaw [\N{DEGREE SIGN}]", xlabel="Time [s]")
    mya.ax[2].minorticks_on()
    mya.ax[2].tick_params(which="minor", length=0)
    mya.ax[2].grid(which="minor", axis="both", linestyle=":", color="0.8")
    mya.ax[2].tick_params(axis="x", which="both", top=True, bottom=True)
    mya.ax[2].tick_params(axis="y", which="both", left=True, right=True)
    mya.f.tight_layout()
    win.NewTab(mya, "Attitude")

    # Position Error
    myperr = MatplotlibWidget(nrows=3, ncols=1, figsize=(10, 6), sharex=True)
    sns.lineplot(x=nav["tR"], y=ned_err[:, 0], color="#a52a2a", ax=myperr.ax[0], label=r"$\mu$")
    sns.lineplot(
        x=nav["tR"], y=3 * np.sqrt(nav["P0"]), color="#100c08", ax=myperr.ax[0], label=r"$3\sigma$"
    )
    sns.lineplot(x=nav["tR"], y=-3 * np.sqrt(nav["P0"]), color="#100c08", ax=myperr.ax[0])
    myperr.ax[0].set(ylabel="N [m]")
    myperr.ax[0].minorticks_on()
    myperr.ax[0].tick_params(which="minor", length=0)
    myperr.ax[0].grid(which="minor", axis="both", linestyle=":", color="0.8")
    myperr.ax[0].tick_params(axis="x", which="both", top=True, bottom=True)
    myperr.ax[0].tick_params(axis="y", which="both", left=True, right=True)
    sns.lineplot(x=nav["tR"], y=ned_err[:, 1], color="#a52a2a", ax=myperr.ax[1])
    sns.lineplot(x=nav["tR"], y=3 * np.sqrt(nav["P1"]), color="#100c08", ax=myperr.ax[1])
    sns.lineplot(x=nav["tR"], y=-3 * np.sqrt(nav["P1"]), color="#100c08", ax=myperr.ax[1])
    myperr.ax[1].set(ylabel="E [m]")
    myperr.ax[1].minorticks_on()
    myperr.ax[1].tick_params(which="minor", length=0)
    myperr.ax[1].grid(which="minor", axis="both", linestyle=":", color="0.8")
    myperr.ax[1].tick_params(axis="x", which="both", top=True, bottom=True)
    myperr.ax[1].tick_params(axis="y", which="both", left=True, right=True)
    sns.lineplot(x=nav["tR"], y=ned_err[:, 2] + 20, color="#a52a2a", ax=myperr.ax[2])
    sns.lineplot(x=nav["tR"], y=3 * np.sqrt(nav["P2"]), color="#100c08", ax=myperr.ax[2])
    sns.lineplot(x=nav["tR"], y=-3 * np.sqrt(nav["P2"]), color="#100c08", ax=myperr.ax[2])
    myperr.ax[2].set(ylabel="D [m]", xlabel="Time [s]")
    myperr.ax[2].minorticks_on()
    myperr.ax[2].tick_params(which="minor", length=0)
    myperr.ax[2].grid(which="minor", axis="both", linestyle=":", color="0.8")
    myperr.ax[2].tick_params(axis="x", which="both", top=True, bottom=True)
    myperr.ax[2].tick_params(axis="y", which="both", left=True, right=True)
    myperr.f.tight_layout()
    win.NewTab(myperr, "Position Error")

    # Velocity Error
    myverr = MatplotlibWidget(nrows=3, ncols=1, figsize=(10, 6), sharex=True)
    sns.lineplot(x=nav["tR"], y=nedv_err[:, 0], color="#a52a2a", ax=myverr.ax[0], label=r"$\mu$")
    sns.lineplot(
        x=nav["tR"], y=3 * np.sqrt(nav["P3"]), color="#100c08", ax=myverr.ax[0], label=r"$3\sigma$"
    )
    sns.lineplot(x=nav["tR"], y=-3 * np.sqrt(nav["P3"]), color="#100c08", ax=myverr.ax[0])
    myverr.ax[0].set(ylabel=r"$\dot{N}$ [m/s]")
    myverr.ax[0].minorticks_on()
    myverr.ax[0].tick_params(which="minor", length=0)
    myverr.ax[0].grid(which="minor", axis="both", linestyle=":", color="0.8")
    myverr.ax[0].tick_params(axis="x", which="both", top=True, bottom=True)
    myverr.ax[0].tick_params(axis="y", which="both", left=True, right=True)
    sns.lineplot(x=nav["tR"], y=nedv_err[:, 1], color="#a52a2a", ax=myverr.ax[1])
    sns.lineplot(x=nav["tR"], y=3 * np.sqrt(nav["P4"]), color="#100c08", ax=myverr.ax[1])
    sns.lineplot(x=nav["tR"], y=-3 * np.sqrt(nav["P4"]), color="#100c08", ax=myverr.ax[1])
    myverr.ax[1].set(ylabel=r"$\dot{E}$ [m/s]")
    myverr.ax[1].minorticks_on()
    myverr.ax[1].tick_params(which="minor", length=0)
    myverr.ax[1].grid(which="minor", axis="both", linestyle=":", color="0.8")
    myverr.ax[1].tick_params(axis="x", which="both", top=True, bottom=True)
    myverr.ax[1].tick_params(axis="y", which="both", left=True, right=True)
    sns.lineplot(x=nav["tR"], y=nedv_err[:, 2], color="#a52a2a", ax=myverr.ax[2])
    sns.lineplot(x=nav["tR"], y=3 * np.sqrt(nav["P5"]), color="#100c08", ax=myverr.ax[2])
    sns.lineplot(x=nav["tR"], y=-3 * np.sqrt(nav["P5"]), color="#100c08", ax=myverr.ax[2])
    myverr.ax[2].set(ylabel=r"$\dot{D}$ [m/s]", xlabel="Time [s]")
    myverr.ax[2].minorticks_on()
    myverr.ax[2].tick_params(which="minor", length=0)
    myverr.ax[2].grid(which="minor", axis="both", linestyle=":", color="0.8")
    myverr.ax[2].tick_params(axis="x", which="both", top=True, bottom=True)
    myverr.ax[2].tick_params(axis="y", which="both", left=True, right=True)
    myverr.f.tight_layout()
    win.NewTab(myverr, "Velocity Error")

    # Attitude Error
    myaerr = MatplotlibWidget(nrows=1, ncols=1, figsize=(10, 6), sharex=True)
    # sns.lineplot(x=nav["tR"], y=rpy_err[:, 0], color="#a52a2a", ax=myaerr.ax[0])
    # myaerr.ax[0].set(ylabel="Roll [\N{DEGREE SIGN}]")
    # myaerr.ax[0].minorticks_on()
    # myaerr.ax[0].tick_params(which="minor", length=0)
    # myaerr.ax[0].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myaerr.ax[0].tick_params(axis="x", which="both", top=True, bottom=True)
    # myaerr.ax[0].tick_params(axis="y", which="both", left=True, right=True)
    # sns.lineplot(x=nav["tR"], y=rpy_err[:, 1], color="#a52a2a", ax=myaerr.ax[1])
    # myaerr.ax[1].set(ylabel="Pitch [\N{DEGREE SIGN}]")
    # myaerr.ax[1].minorticks_on()
    # myaerr.ax[1].tick_params(which="minor", length=0)
    # myaerr.ax[1].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myaerr.ax[1].tick_params(axis="x", which="both", top=True, bottom=True)
    # myaerr.ax[1].tick_params(axis="y", which="both", left=True, right=True)
    sns.lineplot(
        x=nav["tR"], y=rpy_err[:, 2] - 7, color="#a2e3b8", ax=myaerr.ax, label=r"7$^\circ$ Bias"
    )
    sns.lineplot(x=nav["tR"], y=rpy_err[:, 2], color="#a52a2a", ax=myaerr.ax, label=r"$\mu$")
    sns.lineplot(
        x=nav["tR"],
        y=3 * np.sqrt(nav["P8"]) * RAD2DEG,
        color="#100c08",
        ax=myaerr.ax,
        label=r"$3\sigma$",
    )
    sns.lineplot(x=nav["tR"], y=-3 * np.sqrt(nav["P8"]) * RAD2DEG, color="#100c08", ax=myaerr.ax)
    myaerr.ax.set(ylabel="Yaw [\N{DEGREE SIGN}]", xlabel="Time [s]")
    myaerr.ax.minorticks_on()
    myaerr.ax.tick_params(which="minor", length=0)
    myaerr.ax.grid(which="minor", axis="both", linestyle=":", color="0.8")
    myaerr.ax.tick_params(axis="x", which="both", top=True, bottom=True)
    myaerr.ax.tick_params(axis="y", which="both", left=True, right=True)
    myaerr.f.tight_layout()
    win.NewTab(myaerr, "Attitude Error")

    # C/No
    mycno = MatplotlibWidget(figsize=(10, 6))
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
    mydopp = MatplotlibWidget(figsize=(10, 6))
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
    mytrack = MatplotlibWidget(figsize=(10, 6))
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

    # save plots
    if SAVE:
        outdir = Path("/media/daniel/Sturdivant/Thesis-Data/Live-Sky-Plots/")
        outdir.mkdir(parents=True, exist_ok=True)
        myperr.f.savefig(
            outdir / f"live_sky_pos_err.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        myverr.f.savefig(
            outdir / f"live_sky_vel_err.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        myaerr.f.savefig(
            outdir / f"live_sky_yaw_err.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        mya.f.savefig(
            outdir / f"live_sky_att.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        mycno.f.savefig(
            outdir / f"live_sky_cno.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

    win.show()
    sys.exit(app.exec())
