import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt6 import QtWidgets
from cycler import cycler

sys.path.append("scripts")
from utils.parsers import ParseSturdrLogs, ParseNavSimStates
from utils.plotters import MyWindow, FoliumPlotWidget, MatplotlibWidget, SkyPlot
from navtools._navtools_core.attitude import quat2euler
from navtools import RAD2DEG


if __name__ == "__main__":
    # COLORS = ["#100c08", "#a52a2a", "#a2e3b8"]
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
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.default": "it",
        },
        font_scale=1.5,
    )
    color_cycle = list(sns.color_palette().as_hex())
    color_cycle.append("#100c08")  # "#a2e3b8"
    color_cycle = cycler("color", color_cycle)

    # i know these are the satellites in the 'sim_ephem.bin' file
    # fmt: off
    # svid = ["GPS1", "GPS2", "GPS3", "GPS6", "GPS11", "GPS14", "GPS17", "GPS19", "GPS22", "GPS24", "GPS30"]
    # truth = ParseNavSimStates("data/ground_sim.bin")
    svid = ["GPS5", "GPS10", "GPS13", "GPS15", "GPS18", "GPS23", "GPS24", "GPS27", "GPS29", "GPS32"]
    truth = ParseNavSimStates("data/drone_sim.bin")
    # fmt: on

    # parse results
    nav, err, channels = ParseSturdrLogs(
        "/media/daniel/Sturdivant/Thesis-Data/Signal-Sim/drone-sim/CNo_20_dB/Run0", True, True
    )
    # nav, channels = ParseSturdrLogs("./results/Signal-Sim/drone-sim/CNo_40_dB/Run0", True, False)
    nav["t"] = nav["tR"] - nav.loc[0, "tR"] + err.loc[0, "t"]
    nav["P6"] *= RAD2DEG**2
    nav["P7"] *= RAD2DEG**2
    nav["P8"] *= RAD2DEG**2
    # nav1, channels1 = ParseSturdrLogs("./results/Signal-Sim/drone-sim/CNo_40_dB/Run1", True, False)
    # nav1["t"] = nav1["tR"] - nav1.loc[0, "tR"] + 20.2  # + err.loc[0, "t"]
    # nav1["P6"] *= RAD2DEG**2
    # nav1["P7"] *= RAD2DEG**2
    # nav1["P8"] *= RAD2DEG**2

    # create window
    app = QtWidgets.QApplication(sys.argv)
    win = MyWindow("Signal Sim Results")

    # my_pos_err = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    # my_vel_err = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    # my_att_err = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    # for ii in range(1, 6):
    #     nav, err, channels = ParseSturdrLogs(
    #         f"/media/daniel/Sturdivant/Thesis-Data/Signal-Sim/drone-sim/CNo_30_dB/Run{ii}",
    #         True,
    #         True,
    #     )
    #     sns.lineplot(x=err["t"], y=err["N"], ax=my_pos_err.ax[0])
    #     sns.lineplot(x=err["t"], y=err["E"], ax=my_pos_err.ax[1])
    #     sns.lineplot(x=err["t"], y=err["D"], ax=my_pos_err.ax[2])
    #     sns.lineplot(x=err["t"], y=err["vN"], ax=my_vel_err.ax[0])
    #     sns.lineplot(x=err["t"], y=err["vE"], ax=my_vel_err.ax[1])
    #     sns.lineplot(x=err["t"], y=err["vD"], ax=my_vel_err.ax[2])
    #     sns.lineplot(x=err["t"], y=err["Roll"], ax=my_att_err.ax[0])
    #     sns.lineplot(x=err["t"], y=err["Pitch"], ax=my_att_err.ax[1])
    #     sns.lineplot(x=err["t"], y=err["Yaw"], ax=my_att_err.ax[2])
    # win.NewTab(my_pos_err, "Position Error")
    # win.NewTab(my_vel_err, "Velocity Error")
    # win.NewTab(my_att_err, "Attitude Error")

    # # open folium map
    # mymap = FoliumPlotWidget(geobasemap="satellite", zoom=16)
    # mymap.AddLine(
    #     np.rad2deg(nav.loc[::50][["Lat", "Lon"]].values).tolist(),
    #     color="#a52a2a",
    #     weight=5,
    #     opacity=1,
    # )
    # # mymap.AddLine(
    # #     np.rad2deg(nav1.loc[::50][["Lat", "Lon"]].values).tolist(),
    # #     color="#100c08",
    # #     weight=5,
    # #     opacity=1,
    # # )
    # mymap.AddLine(
    #     truth.loc[::100][["lat", "lon"]].values.tolist(), color="#a2e3b8", weight=1, opacity=1
    # )
    # mymap.AddLegend({"EKF": "#a52a2a", "Truth": "#a2e3b8"})
    # # mymap.AddLegend({"DPD": "#a52a2a", "M+W": "#100c08", "Truth": "#a2e3b8"})
    # win.NewTab(mymap, "GeoPlot")

    # # plot channel data
    # myv = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    # sns.lineplot(x=nav["t"], y=nav["vN"], label="EKF", color="#a52a2a", ax=myv.ax[0])
    # sns.lineplot(
    #     x=truth["t"] / 1000.0, y=truth.loc[:, "vn"], label="Truth", color="#100c08", ax=myv.ax[0]
    # )
    # myv.ax[0].set(ylabel="North [m/s]")
    # myv.ax[0].minorticks_on()
    # myv.ax[0].tick_params(which="minor", length=0)
    # myv.ax[0].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myv.ax[0].tick_params(axis="x", which="both", top=True, bottom=True)
    # myv.ax[0].tick_params(axis="y", which="both", left=True, right=True)
    # myv.ax[0].legend(loc="upper left")
    # sns.lineplot(x=nav["t"], y=nav["vE"], color="#a52a2a", ax=myv.ax[1])
    # sns.lineplot(x=truth["t"] / 1000.0, y=truth.loc[:, "ve"], color="#100c08", ax=myv.ax[1])
    # myv.ax[1].set(ylabel="East [m/s]")
    # myv.ax[1].minorticks_on()
    # myv.ax[1].tick_params(which="minor", length=0)
    # myv.ax[1].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myv.ax[1].tick_params(axis="x", which="both", top=True, bottom=True)
    # myv.ax[1].tick_params(axis="y", which="both", left=True, right=True)
    # sns.lineplot(x=nav["t"], y=nav["vD"], color="#a52a2a", ax=myv.ax[2])
    # sns.lineplot(x=truth["t"] / 1000.0, y=truth.loc[:, "vd"], color="#100c08", ax=myv.ax[2])
    # myv.ax[2].set(ylabel="Down [m/s]", xlabel="Time [s]")
    # myv.ax[2].minorticks_on()
    # myv.ax[2].tick_params(which="minor", length=0)
    # myv.ax[2].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myv.ax[2].tick_params(axis="x", which="both", top=True, bottom=True)
    # myv.ax[2].tick_params(axis="y", which="both", left=True, right=True)
    # myv.f.tight_layout()
    # win.NewTab(myv, "Velocity")

    mya = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    rpy = np.zeros((len(nav), 3), order="F")
    for ii in range(len(nav)):
        q = np.array(
            [nav.loc[ii, "qw"], nav.loc[ii, "qx"], nav.loc[ii, "qy"], nav.loc[ii, "qz"]], order="F"
        )
        rpy[ii, :] = np.rad2deg(quat2euler(q, True))
    # rpy1 = np.zeros((len(nav1), 3), order="F")
    # for ii in range(len(nav1)):
    #     q = np.array(
    #         [nav1.loc[ii, "qw"], nav1.loc[ii, "qx"], nav1.loc[ii, "qy"], nav1.loc[ii, "qz"]],
    #         order="F",
    #     )
    #     rpy1[ii, :] = np.rad2deg(quat2euler(q, True))
    # sns.lineplot(x=nav["t"], y=rpy[:, 0], color="#a52a2a", ax=mya.ax[0], label="$\\Delta \\varphi$")
    # sns.lineplot(x=nav1["t"], y=rpy1[:, 0], color="#100c08", ax=mya.ax[0], label="MUSIC + Wahba's")
    # sns.lineplot(
    #     x=truth.loc[2020:, "t"] / 1000.0,
    #     y=truth.loc[:, "r"],
    #     linestyle="--",
    #     color="#a2e3b8",
    #     linewidth=1,
    #     ax=mya.ax[0],
    #     label="Truth",
    # )
    sns.lineplot(x=nav["t"], y=rpy[:, 0], label="EKF", color="#a52a2a", ax=mya.ax[0])
    sns.lineplot(
        x=truth["t"] / 1000.0, y=truth.loc[:, "r"], label="Truth", color="#100c08", ax=mya.ax[0]
    )
    mya.ax[0].set(ylabel="Roll [\N{DEGREE SIGN}]")
    mya.ax[0].minorticks_on()
    mya.ax[0].tick_params(which="minor", length=0)
    mya.ax[0].grid(which="minor", axis="both", linestyle=":", color="0.8")
    mya.ax[0].tick_params(axis="x", which="both", top=True, bottom=True)
    mya.ax[0].tick_params(axis="y", which="both", left=True, right=True)
    mya.ax[0].legend(loc="upper left")
    # sns.lineplot(x=nav["t"], y=rpy[:, 1], color="#a52a2a", ax=mya.ax[1])
    # sns.lineplot(x=nav1["t"], y=rpy1[:, 1], color="#100c08", ax=mya.ax[1])
    # sns.lineplot(
    #     x=truth.loc[2020:, "t"] / 1000.0,
    #     y=truth.loc[:, "p"],
    #     linestyle="--",
    #     color="#a2e3b8",
    #     linewidth=1,
    #     ax=mya.ax[1],
    # )
    sns.lineplot(x=nav["t"], y=rpy[:, 1], color="#a52a2a", ax=mya.ax[1])
    sns.lineplot(x=truth["t"] / 1000.0, y=truth.loc[:, "p"], color="#100c08", ax=mya.ax[1])
    mya.ax[1].set(ylabel="Pitch [\N{DEGREE SIGN}]")
    mya.ax[1].minorticks_on()
    mya.ax[1].tick_params(which="minor", length=0)
    mya.ax[1].grid(which="minor", axis="both", linestyle=":", color="0.8")
    mya.ax[1].tick_params(axis="x", which="both", top=True, bottom=True)
    mya.ax[1].tick_params(axis="y", which="both", left=True, right=True)
    # sns.lineplot(x=nav["t"], y=rpy[:, 2], color="#a52a2a", ax=mya.ax[2])
    # sns.lineplot(x=nav1["t"], y=rpy1[:, 2], color="#100c08", ax=mya.ax[2])
    # sns.lineplot(
    #     x=truth.loc[2020:, "t"] / 1000.0,
    #     y=truth.loc[:, "y"],
    #     linestyle="--",
    #     color="#a2e3b8",
    #     linewidth=1,
    #     ax=mya.ax[2],
    # )
    sns.lineplot(x=nav["t"], y=rpy[:, 2], color="#a52a2a", ax=mya.ax[2])
    sns.lineplot(x=truth["t"] / 1000.0, y=truth.loc[:, "y"], color="#100c08", ax=mya.ax[2])
    mya.ax[2].set(ylabel="Yaw [\N{DEGREE SIGN}]", xlabel="Time [s]")
    mya.ax[2].minorticks_on()
    mya.ax[2].tick_params(which="minor", length=0)
    mya.ax[2].grid(which="minor", axis="both", linestyle=":", color="0.8")
    mya.ax[2].tick_params(axis="x", which="both", top=True, bottom=True)
    mya.ax[2].tick_params(axis="y", which="both", left=True, right=True)
    mya.f.tight_layout()
    win.NewTab(mya, "Attitude")
    # mya.f.savefig("./results/signal-sim-att-difference.pdf", format="pdf")

    # myperr = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    # sns.lineplot(x=err["t"], y=err["N"], label="$\\mu$", color="#100c08", ax=myperr.ax[0])
    # sns.lineplot(
    #     x=nav["t"], y=3 * np.sqrt(nav["P0"]), label="$3\\sigma$", color="#a52a2a", ax=myperr.ax[0]
    # )
    # sns.lineplot(x=nav["t"], y=-3 * np.sqrt(nav["P0"]), color="#a52a2a", ax=myperr.ax[0])
    # myperr.ax[0].set(ylabel="North [m]")  # ylim=[-3, 3],
    # myperr.ax[0].minorticks_on()
    # myperr.ax[0].tick_params(which="minor", length=0)
    # myperr.ax[0].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myperr.ax[0].tick_params(axis="x", which="both", top=True, bottom=True)
    # myperr.ax[0].tick_params(axis="y", which="both", left=True, right=True)
    # myperr.ax[0].legend(loc="upper right")
    # sns.lineplot(x=err["t"], y=err["E"], color="#100c08", ax=myperr.ax[1])
    # sns.lineplot(x=nav["t"], y=3 * np.sqrt(nav["P1"]), color="#a52a2a", ax=myperr.ax[1])
    # sns.lineplot(x=nav["t"], y=-3 * np.sqrt(nav["P1"]), color="#a52a2a", ax=myperr.ax[1])
    # myperr.ax[1].set(ylabel="East [m]")  # ylim=[-3, 3],
    # myperr.ax[1].minorticks_on()
    # myperr.ax[1].tick_params(which="minor", length=0)
    # myperr.ax[1].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myperr.ax[1].tick_params(axis="x", which="both", top=True, bottom=True)
    # myperr.ax[1].tick_params(axis="y", which="both", left=True, right=True)
    # sns.lineplot(x=err["t"], y=err["D"], color="#100c08", ax=myperr.ax[2])
    # sns.lineplot(x=nav["t"], y=3 * np.sqrt(nav["P2"]), color="#a52a2a", ax=myperr.ax[2])
    # sns.lineplot(x=nav["t"], y=-3 * np.sqrt(nav["P2"]), color="#a52a2a", ax=myperr.ax[2])
    # myperr.ax[2].set(ylabel="Down [m]", xlabel="Time [s]")  # ylim=[-6, 6],
    # myperr.ax[2].minorticks_on()
    # myperr.ax[2].tick_params(which="minor", length=0)
    # myperr.ax[2].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myperr.ax[2].tick_params(axis="x", which="both", top=True, bottom=True)
    # myperr.ax[2].tick_params(axis="y", which="both", left=True, right=True)
    # myperr.f.tight_layout()
    # win.NewTab(myperr, "Position Error")

    # myverr = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    # sns.lineplot(x=err["t"], y=err["vN"], label="$\\mu$", color="#100c08", ax=myverr.ax[0])
    # sns.lineplot(
    #     x=nav["t"], y=3 * np.sqrt(nav["P3"]), label="$3\\sigma$", color="#a52a2a", ax=myverr.ax[0]
    # )
    # sns.lineplot(x=nav["t"], y=-3 * np.sqrt(nav["P3"]), color="#a52a2a", ax=myverr.ax[0])
    # myverr.ax[0].set(ylabel="North [m/s]")  # ylim=[-3, 3],
    # myverr.ax[0].minorticks_on()
    # myverr.ax[0].tick_params(which="minor", length=0)
    # myverr.ax[0].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myverr.ax[0].tick_params(axis="x", which="both", top=True, bottom=True)
    # myverr.ax[0].tick_params(axis="y", which="both", left=True, right=True)
    # myverr.ax[0].legend(loc="upper right")
    # sns.lineplot(x=err["t"], y=err["vE"], color="#100c08", ax=myverr.ax[1])
    # sns.lineplot(x=nav["t"], y=3 * np.sqrt(nav["P4"]), color="#a52a2a", ax=myverr.ax[1])
    # sns.lineplot(x=nav["t"], y=-3 * np.sqrt(nav["P4"]), color="#a52a2a", ax=myverr.ax[1])
    # myverr.ax[1].set(ylabel="East [m/s]")  # ylim=[-3, 3],
    # myverr.ax[1].minorticks_on()
    # myverr.ax[1].tick_params(which="minor", length=0)
    # myverr.ax[1].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myverr.ax[1].tick_params(axis="x", which="both", top=True, bottom=True)
    # myverr.ax[1].tick_params(axis="y", which="both", left=True, right=True)
    # sns.lineplot(x=err["t"], y=err["vD"], color="#100c08", ax=myverr.ax[2])
    # sns.lineplot(x=nav["t"], y=3 * np.sqrt(nav["P5"]), color="#a52a2a", ax=myverr.ax[2])
    # sns.lineplot(x=nav["t"], y=-3 * np.sqrt(nav["P5"]), color="#a52a2a", ax=myverr.ax[2])
    # myverr.ax[2].set(ylabel="Down [m/s]", xlabel="Time [s]")  # ylim=[-6, 6],
    # myverr.ax[2].minorticks_on()
    # myverr.ax[2].tick_params(which="minor", length=0)
    # myverr.ax[2].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myverr.ax[2].tick_params(axis="x", which="both", top=True, bottom=True)
    # myverr.ax[2].tick_params(axis="y", which="both", left=True, right=True)
    # myverr.f.tight_layout()
    # win.NewTab(myverr, "Velocity Error")

    # myaerr = MatplotlibWidget(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    # sns.lineplot(x=err["t"], y=err["Roll"], label="$\\mu$", color="#100c08", ax=myaerr.ax[0])
    # # sns.lineplot(x=err["t"], y=truth.iloc[2:-2:2]["r"] - rpy[:, 0], color="g", ax=myaerr.ax[0])
    # sns.lineplot(
    #     x=nav["t"], y=3 * np.sqrt(nav["P6"]), label="$3\\sigma$", color="#a52a2a", ax=myaerr.ax[0]
    # )
    # sns.lineplot(x=nav["t"], y=-3 * np.sqrt(nav["P6"]), color="#a52a2a", ax=myaerr.ax[0])
    # myaerr.ax[0].set(ylabel="Roll [\N{DEGREE SIGN}]")  # ylim=[-3, 3],
    # myaerr.ax[0].minorticks_on()
    # myaerr.ax[0].tick_params(which="minor", length=0)
    # myaerr.ax[0].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myaerr.ax[0].tick_params(axis="x", which="both", top=True, bottom=True)
    # myaerr.ax[0].tick_params(axis="y", which="both", left=True, right=True)
    # myaerr.ax[0].legend(loc="upper right")
    # sns.lineplot(x=err["t"], y=err["Pitch"], color="#100c08", ax=myaerr.ax[1])
    # # sns.lineplot(x=err["t"], y=truth.iloc[2:-2:2]["p"] - rpy[:, 1], color="g", ax=myaerr.ax[1])
    # sns.lineplot(x=nav["t"], y=3 * np.sqrt(nav["P7"]), color="#a52a2a", ax=myaerr.ax[1])
    # sns.lineplot(x=nav["t"], y=-3 * np.sqrt(nav["P7"]), color="#a52a2a", ax=myaerr.ax[1])
    # myaerr.ax[1].set(ylabel="Pitch [\N{DEGREE SIGN}]")  # ylim=[-3, 3],
    # myaerr.ax[1].minorticks_on()
    # myaerr.ax[1].tick_params(which="minor", length=0)
    # myaerr.ax[1].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myaerr.ax[1].tick_params(axis="x", which="both", top=True, bottom=True)
    # myaerr.ax[1].tick_params(axis="y", which="both", left=True, right=True)
    # sns.lineplot(x=err["t"], y=err["Yaw"], color="#100c08", ax=myaerr.ax[2])
    # # sns.lineplot(x=err["t"], y=truth.iloc[2:-2:2]["y"] - rpy[:, 2], color="g", ax=myaerr.ax[2])
    # sns.lineplot(x=nav["t"], y=3 * np.sqrt(nav["P8"]), color="#a52a2a", ax=myaerr.ax[2])
    # sns.lineplot(x=nav["t"], y=-3 * np.sqrt(nav["P8"]), color="#a52a2a", ax=myaerr.ax[2])
    # myaerr.ax[2].set(ylabel="Yaw [\N{DEGREE SIGN}]", xlabel="Time [s]")  # ylim=[-6, 6],
    # myaerr.ax[2].minorticks_on()
    # myaerr.ax[2].tick_params(which="minor", length=0)
    # myaerr.ax[2].grid(which="minor", axis="both", linestyle=":", color="0.8")
    # myaerr.ax[2].tick_params(axis="x", which="both", top=True, bottom=True)
    # myaerr.ax[2].tick_params(axis="y", which="both", left=True, right=True)
    # myaerr.f.tight_layout()
    # win.NewTab(myaerr, "Attitude Error")

    # mycno = MatplotlibWidget(figsize=(8, 8))
    # mycno.ax.set_prop_cycle(color_cycle)
    # for i in range(len(channels)):
    #     sns.lineplot(x=channels[i]["t"] / 1000, y=channels[i]["CNo"], label=svid[i], ax=mycno.ax)
    # mycno.ax.set(xlabel="t [s]", ylabel="C/N$_0$ [dB-Hz]")
    # mycno.ax.minorticks_on()
    # mycno.ax.tick_params(which="minor", length=0)
    # mycno.ax.grid(which="minor", axis="both", linestyle=":", color="0.8")
    # mycno.ax.tick_params(axis="x", which="both", top=True, bottom=True)
    # mycno.ax.tick_params(axis="y", which="both", left=True, right=True)
    # mycno.f.tight_layout()
    # win.NewTab(mycno, "C/No")

    # mycorr = MatplotlibWidget(figsize=(8, 8))
    # mycorr.ax.set_prop_cycle(color_cycle)
    # for i in range(len(channels)):
    #     sns.lineplot(
    #         x=channels[i]["t"],
    #         y=np.sqrt(channels[i]["IP"] ** 2 + channels[i]["QP"] ** 2),
    #         label=svid[i],
    #         ax=mycorr.ax,
    #     )
    #     # sns.lineplot(
    #     #     x=channels[i]["t"],
    #     #     y=channels[i]["IP_reg_0"] ** 2 + channels[i]["QP_reg_0"] ** 2,
    #     #     label=None,
    #     #     ax=mycorr.ax,
    #     # )
    # mycorr.ax.set(xlabel="t [s]", ylabel="Correlator Power")
    # mycorr.ax.minorticks_on()
    # mycorr.ax.tick_params(which="minor", length=0)
    # mycorr.ax.grid(which="minor", axis="both", linestyle=":", color="0.8")
    # mycorr.ax.tick_params(axis="x", which="both", top=True, bottom=True)
    # mycorr.ax.tick_params(axis="y", which="both", left=True, right=True)
    # mycorr.f.tight_layout()
    # win.NewTab(mycorr, "Correlator Power")

    # mypolar = MatplotlibWidget(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    # mypolar.ax.set_prop_cycle(color_cycle)
    # for i in range(len(channels)):
    #     mypolar.ax = SkyPlot(
    #         channels[i]["az"].values, channels[i]["el"].values, [svid[i]], ax=mypolar.ax
    #     )

    # win.NewTab(mypolar, "SkyPlot")

    # open plots
    # plt.show()
    win.show()
    sys.exit(app.exec())
