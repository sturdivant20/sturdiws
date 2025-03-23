import numpy as np
import pandas as pd
from pathlib import Path
from struct import pack
from navtools import PI, TWO_PI, LIGHT_SPEED, RAD2DEG, DEG2RAD, WGS84_E2, WGS84_R0, frames, attitude
from satutils import GPS_L1_FREQUENCY, GPS_CA_CODE_RATE, ephemeris, atmosphere
from navsim import ClockSim, CorrelatorSim, CnoSim, ObservableSim, NormalDistribution, RandomEngine
from sturdins import navsense, KinematicNav
from sturdr import discriminator, lockdetectors


def vt(
    conf: dict,
    eph: np.ndarray[ephemeris.KeplerElements],
    atm: np.ndarray[atmosphere.KlobucharElements],
    truth: pd.DataFrame,
    seed: int = 1,
    run_num: int = 1,
) -> None:
    """
    VT
    ==

    Simulates the correlators and NCOs of a GPS L1 C/A vector tracking receiver. It makes use of the
    common transmit time (CTT) such that the measurements from each satellite can be utilized
    synchronously. This significantly simplifies the simulation by allowing the satellite transmit
    times (tT) to all be synchronized to the time of week (ToW) but allows the NCOs to adjust the
    perceived receive times (tR) of the signals.

    Inputs
    ------
    conf : dict
        Contains the configuration parameters for the simulation
            - scenario : str
            - data_file : str
            - ephem_file : str
            - out_folder : str
            - n_runs : int
            - clock_model : str
            - vel_process_std : float
            - att_process_std : float
            - sec_to_skip : float
            - init_tow : float
            - intmd_freq : float
            - sim_dt : float
            - meas_dt : float
            - tap_epl : float
            - init_cb : float
            - init_cd : float
            - init_cov : list
            - add_init_err : bool
            - jammer_modulation : str
            - jammer_type : str
            - j2s : float
            - is_multi_antenna : bool
            - n_ant : int
            - ant_xyz_0 : float
            - ant_xyz_1 : float
            - ant_xyz_2 : float
            - ant_xyz_3 : float

    eph : pd.DataFrame
        Contains the satellite ephemeris for each satellite
            - iode : float
            - iodc : float
            - toe : float
            - toc : float
            - tgd : float
            - af2 : float
            - af1 : float
            - af0 : float
            - e : float
            - sqrtA : float
            - deltan : float
            - m0 : float
            - omega0 : float
            - omega : float
            - omegaDot : float
            - i0 : float
            - iDot : float
            - cuc : float
            - cus : float
            - cic : float
            - cis : float
            - crc : float
            - crs : float
            - ura : float
            - health : float

    truth : pd.DataFrame
        Contains the truth trajectory information
            t : float
            lat : float
            lon : float
            h : float
            vn : float
            ve : float
            vd : float
            r : float
            p : float
            y : float
    """

    # constants
    M = len(eph)
    lamb = LIGHT_SPEED / GPS_L1_FREQUENCY
    beta = LIGHT_SPEED / GPS_CA_CODE_RATE
    kappa = GPS_CA_CODE_RATE / GPS_L1_FREQUENCY
    r2d_sq = RAD2DEG * RAD2DEG
    J2S = 10 ** (conf["j2s"] / 10) * np.ones(M, dtype=np.double, order="F")
    k_start = int(conf["sec_to_skip"] / conf["sim_dt"])

    # noise generation
    dist = NormalDistribution(0.0, 1.0)
    eng = RandomEngine(seed)

    # receiver sensors
    clock_model = navsense.GetNavClock(conf["clock_model"])
    corr_sim = CorrelatorSim(conf["tap_epl"], M, 2, eng, dist)
    clk_sim = ClockSim(
        clock_model.h0,
        clock_model.h1,
        clock_model.h2,
        conf["init_cb"],
        conf["init_cd"],
        conf["sim_dt"],
        conf["add_init_err"],
        eng,
        dist,
    )
    cno_sim = CnoSim(M, lamb, GPS_CA_CODE_RATE, conf["jammer_modulation"], conf["jammer_type"])
    obs_sim = ObservableSim(eph, atm)
    t_sim = 0.0

    # initialize kalman filter
    init_lla = np.array(
        [truth["lat"][k_start] * DEG2RAD, truth["lon"][k_start] * DEG2RAD, truth["h"][k_start]]
    )
    init_vel = np.array([truth["vn"][k_start], truth["ve"][k_start], truth["vd"][k_start]])
    init_rpy = np.array(
        [
            truth["r"][k_start] * DEG2RAD,
            truth["p"][k_start] * DEG2RAD,
            truth["y"][k_start] * DEG2RAD,
        ]
    )
    init_cb = conf["init_cb"]
    init_cd = conf["init_cd"]
    if conf["add_init_err"]:
        t = 1.0 - WGS84_E2 * np.sin(init_lla[0]) ** 2
        sqt = np.sqrt(t)
        Re = WGS84_R0 / sqt
        Rn = WGS84_R0 * (1 - WGS84_E2) / (t * t / sqt)
        T_r_p = np.diag(
            [1 / (Rn + init_lla[2]), 1 / ((Re + init_lla[2]) * np.cos(init_lla[0])), -1]
        )
        sqrtP = np.diag(conf["init_P"])
        init_lla += T_r_p @ sqrtP[0:3, 0:3] @ np.random.randn(3)
        init_vel += sqrtP[3:6, 3:6] @ np.random.randn(3)
        init_vel += sqrtP[6:9, 6:9] @ np.random.randn(3)
        init_cb += sqrtP[9, 9] @ np.random.randn()
        init_cd += sqrtP[10, 10] @ np.random.randn()
    kf = KinematicNav(
        init_lla[0],
        init_lla[1],
        init_lla[2],
        init_vel[0],
        init_vel[1],
        init_vel[2],
        init_rpy[0],
        init_rpy[1],
        init_rpy[2],
        init_cb,
        init_cd,
    )
    kf.SetClockSpec(clock_model.h0, clock_model.h1, clock_model.h2)
    kf.SetProcessNoise(conf["vel_process_psd"], conf["att_process_psd"])

    # initialize tracking states
    rng = np.zeros(M, dtype=np.double, order="F")
    empty2 = np.zeros(M, dtype=np.double, order="F")
    u = np.zeros((3, M), dtype=np.double, order="F")
    true_state = {
        "psr": np.zeros(M, dtype=np.double, order="F"),
        "psrdot": np.zeros(M, dtype=np.double, order="F"),
        "chip": np.zeros(M, dtype=np.double, order="F"),
        "chiprate": np.zeros(M, dtype=np.double, order="F"),
        "phase": np.zeros(M, dtype=np.double, order="F"),
        "omega": np.zeros(M, dtype=np.double, order="F"),
        "sv_clk": np.zeros((3, M), dtype=np.double, order="F"),
        "sv_pos": np.zeros((3, M), dtype=np.double, order="F"),
        "sv_vel": np.zeros((3, M), dtype=np.double, order="F"),
        "ToW": conf["init_tow"] * np.ones(M, dtype=np.double, order="F"),
    }
    nco_state = {
        "psr": np.zeros(M, dtype=np.double, order="F"),
        "psrdot": np.zeros(M, dtype=np.double, order="F"),
        "chip": np.zeros(M, dtype=np.double, order="F"),
        "chiprate": np.zeros(M, dtype=np.double, order="F"),
        "phase": np.zeros(M, dtype=np.double, order="F"),
        "omega": np.zeros(M, dtype=np.double, order="F"),
        "tT": conf["init_tow"] * np.ones(M, dtype=np.double, order="F"),
        "tR": np.zeros(M, dtype=np.double, order="F"),
        "sv_clk": np.zeros((3, M), dtype=np.double, order="F"),
        "sv_pos": np.zeros((3, M), dtype=np.double, order="F"),
        "sv_vel": np.zeros((3, M), dtype=np.double, order="F"),
    }

    # true state
    true_cb, true_cd = clk_sim.GetCurrentState()
    lla_true = np.array(
        [truth["lat"][k_start] * DEG2RAD, truth["lon"][k_start] * DEG2RAD, truth["h"][k_start]],
        order="F",
    )
    xyz_true = frames.lla2ecef(lla_true)
    xyzv_true = frames.ned2ecefv(
        np.array([truth["vn"][k_start], truth["ve"][k_start], truth["vd"][k_start]], order="F"),
        lla_true,
    )
    obs_sim.GetRangeAndRate(
        true_state["ToW"],
        xyz_true,
        xyzv_true,
        true_cb,
        true_cd,
        u,
        true_state["sv_clk"],
        true_state["sv_pos"],
        true_state["sv_vel"],
        rng,
        true_state["psr"],
        true_state["psrdot"],
        empty2,
    )
    true_state["phase"] = TWO_PI * (conf["intmd_freq"] * t_sim - true_state["psr"] / lamb)
    true_state["omega"] = TWO_PI * (conf["intmd_freq"] - true_state["psrdot"] / lamb)
    true_state["chip"] = true_state["psr"] / beta
    true_state["chiprate"] = GPS_CA_CODE_RATE - kappa * true_state["psrdot"] / lamb

    # nco state
    lla_nco = np.array([kf.phi_, kf.lam_, kf.h_], order="F")
    xyz_nco = frames.lla2ecef(lla_nco)
    xyzv_nco = frames.ned2ecefv(np.array([kf.vn_, kf.ve_, kf.vd_], order="F"), lla_nco)
    obs_sim.GetRangeAndRate(
        nco_state["tT"],
        xyz_nco,
        xyzv_nco,
        kf.cb_,
        kf.cd_,
        u,
        nco_state["sv_clk"],
        nco_state["sv_pos"],
        nco_state["sv_vel"],
        rng,
        nco_state["psr"],
        nco_state["psrdot"],
        empty2,
    )
    nco_state["phase"] = TWO_PI * (conf["intmd_freq"] * t_sim - nco_state["psr"] / lamb)
    nco_state["omega"] = TWO_PI * (conf["intmd_freq"] - nco_state["psrdot"] / lamb)
    nco_state["chip"] = nco_state["psr"] / beta
    nco_state["chiprate"] = GPS_CA_CODE_RATE - kappa * nco_state["psrdot"] / lamb
    nco_state["tR"] = nco_state["tT"] + nco_state["psr"] / LIGHT_SPEED

    # initialize cno/discriminator estimators
    cno_state = {
        "alpha": 0.0025,
        "est_cno": 1000 * np.ones(M, dtype=np.double, order="F"),  # 25119
        "true_cno": np.zeros(M, dtype=np.double, order="F"),
        "detectors": [lockdetectors.LockDetectors(0.0025) for _ in range(M)],
    }
    dR = np.zeros(M, order="F")
    dRR = np.zeros(M, order="F")
    dR_var = np.zeros(M, order="F")
    dRR_var = np.zeros(M, order="F")

    # open files for binary output
    mypath = Path(conf["out_folder"]) / conf["scenario"] / str(run_num)
    mypath.mkdir(parents=True, exist_ok=True)
    nav_file = open(mypath / "Nav_Results_Log.bin", "wb")
    err_file = open(mypath / "Err_Results_Log.bin", "wb")
    var_file = open(mypath / "Var_Results_Log.bin", "wb")
    chn_files = [open(mypath / f"Channel_{i}_Results_Log.bin", "wb") for i in range(M)]

    # run simulation
    k_iq = 1
    k_meas = 1
    k_corr = 1
    dk_meas = int(conf["meas_dt"] / conf["sim_dt"])
    dk_corr = int(dk_meas / 2)
    kf.Propagate(conf["meas_dt"])
    for k_sim in range(k_start + 1, len(truth)):
        true_cb, true_cd = clk_sim.Simulate()
        t_sim += conf["sim_dt"]

        # propagate truth states
        true_state["ToW"] += conf["sim_dt"]
        old_chip = true_state["chip"]
        lla_true = np.array(
            [truth["lat"][k_sim] * DEG2RAD, truth["lon"][k_sim] * DEG2RAD, truth["h"][k_sim]],
            order="F",
        )
        xyz_true = frames.lla2ecef(lla_true)
        xyzv_true = frames.ned2ecefv(
            np.array([truth["vn"][k_sim], truth["ve"][k_sim], truth["vd"][k_sim]], order="F"),
            lla_true,
        )
        obs_sim.GetRangeAndRate(
            true_state["ToW"],
            xyz_true,
            xyzv_true,
            true_cb,
            true_cd,
            u,
            true_state["sv_clk"],
            true_state["sv_pos"],
            true_state["sv_vel"],
            rng,
            true_state["psr"],
            true_state["psrdot"],
            empty2,
        )
        true_state["phase"] = TWO_PI * (conf["intmd_freq"] * t_sim - true_state["psr"] / lamb)
        true_state["omega"] = TWO_PI * (conf["intmd_freq"] - true_state["psrdot"] / lamb)
        true_state["chip"] = true_state["psr"] / beta
        true_state["chiprate"] = GPS_CA_CODE_RATE - (true_state["chip"] - old_chip) / conf["sim_dt"]

        # propagate nco states
        t_int = (GPS_CA_CODE_RATE / nco_state["chiprate"]) * conf["sim_dt"]
        nco_state["tT"] += conf["sim_dt"]
        nco_state["tR"] += t_int
        nco_state["chip"] = (nco_state["tR"] - nco_state["tT"]) * GPS_CA_CODE_RATE
        nco_state["phase"] += nco_state["omega"] * t_int

        # get true cno
        cno_state["true_cno"] = cno_sim.FsplPlusJammerModel(J2S, rng)

        # update correlators
        corr_sim.NextSample(
            conf["sim_dt"],
            cno_state["true_cno"],
            true_state["chip"],
            true_state["chiprate"],
            true_state["phase"],
            true_state["omega"],
            nco_state["chip"],
            nco_state["chiprate"],
            nco_state["phase"],
            nco_state["omega"],
        )

        # extract correlators
        if k_corr == dk_corr:
            if k_iq == 1:
                R1 = corr_sim.GetCorrelators()
                k_iq = 2
            else:
                R2 = corr_sim.GetCorrelators()
                k_iq = 1
            k_corr = 0

        # measurement update
        if k_meas == dk_meas:
            # complete integration
            R = R1 + R2

            # update cno/discriminators
            for i in range(M):
                cno_state["detectors"][i].Update(R[1, i], conf["meas_dt"])
                cno_state["est_cno"][i] = cno_state["detectors"][i].GetCno()
                dR[i] = beta * discriminator.DllNneml(R[0, i], R[2, i])
                dRR[i] = (
                    -lamb / TWO_PI * discriminator.FllAtan2(R1[1, i], R2[1, i], conf["meas_dt"])
                )
                dR_var[i] = beta**2 * discriminator.DllVariance(
                    cno_state["est_cno"][i], conf["meas_dt"]
                )
                dRR_var[i] = (lamb / TWO_PI) ** 2 * discriminator.FllVariance(
                    cno_state["est_cno"][i], conf["meas_dt"]
                )

            # get satellite and measurement information
            lla_nco = np.array([kf.phi_, kf.lam_, kf.h_], order="F")
            xyz_nco = frames.lla2ecef(lla_nco)
            xyzv_nco = frames.ned2ecefv(np.array([kf.vn_, kf.ve_, kf.vd_], order="F"), lla_nco)
            obs_sim.GetRangeAndRate(
                nco_state["tT"],
                xyz_nco,
                xyzv_nco,
                kf.cb_,
                kf.cd_,
                u,
                nco_state["sv_clk"],
                nco_state["sv_pos"],
                nco_state["sv_vel"],
                rng,
                nco_state["psr"],
                nco_state["psrdot"],
                empty2,
            )

            # kalman update
            psr_meas = (nco_state["tR"] - nco_state["tT"]) * LIGHT_SPEED + dR
            psrdot_meas = -lamb * (nco_state["omega"] / TWO_PI - conf["intmd_freq"]) + dRR
            kf.GnssUpdate(
                nco_state["sv_pos"], nco_state["sv_vel"], psr_meas, psrdot_meas, dR_var, dRR_var
            )

            # save to binary files
            rpy_nco = attitude.dcm2euler(kf.C_b_l_, True) * RAD2DEG
            lla_nco = np.array([kf.phi_, kf.lam_, kf.h_], order="F")
            U_NED = frames.ecef2nedDcm(lla_nco) @ u
            data = [
                truth["t"][k_sim],
                nco_state["tT"][0],
                kf.phi_ * RAD2DEG,
                kf.lam_ * RAD2DEG,
                kf.h_,
                kf.vn_,
                kf.ve_,
                kf.vd_,
                rpy_nco[0],
                rpy_nco[1],
                rpy_nco[2],
                kf.cb_,
                kf.cd_,
            ]
            nav_file.write(pack("d" * 13, *data))
            ned_err = frames.lla2ned(lla_nco, lla_true)
            data = [
                truth["t"][k_sim],
                nco_state["tT"][0],
                ned_err[0],
                ned_err[1],
                ned_err[2],
                truth["vn"][k_sim] - kf.vn_,
                truth["ve"][k_sim] - kf.ve_,
                truth["vd"][k_sim] - kf.vd_,
                0.0,
                0.0,
                0.0,
                true_cb - kf.cb_,
                true_cd - kf.cd_,
            ]
            err_file.write(pack("d" * 13, *data))
            data = [
                truth["t"][k_sim],
                nco_state["tT"][0],
                kf.P_[0, 0],
                kf.P_[1, 1],
                kf.P_[2, 2],
                kf.P_[3, 3],
                kf.P_[4, 4],
                kf.P_[5, 5],
                kf.P_[6, 6] * r2d_sq,
                kf.P_[7, 7] * r2d_sq,
                kf.P_[8, 8] * r2d_sq,
                kf.P_[9, 9],
                kf.P_[10, 10],
            ]
            var_file.write(pack("d" * 13, *data))
            for i in range(M):
                data = [
                    truth["t"][k_sim],
                    true_state["ToW"][i],
                    RAD2DEG * np.atan2(u[1, i], u[0, i]),
                    RAD2DEG * -np.asin(u[2, i]),
                    true_state["phase"][i],
                    true_state["omega"][i],
                    true_state["chip"][i],
                    true_state["chiprate"][i],
                    10 * np.log10(cno_state["true_cno"][i]),
                    nco_state["phase"][i],
                    nco_state["omega"][i],
                    nco_state["chip"][i],
                    nco_state["chiprate"][i],
                    10 * np.log10(cno_state["est_cno"][i]),
                    R[0, i].real,
                    R[0, i].imag,
                    R[1, i].real,
                    R[1, i].imag,
                    R[2, i].real,
                    R[2, i].imag,
                    R1[1, i].real,
                    R1[1, i].imag,
                    R2[1, i].real,
                    R2[1, i].imag,
                ]
                chn_files[i].write(pack("d" * 24, *data))

            # propagate
            kf.Propagate(conf["meas_dt"])
            lla_nco = np.array([kf.phi_, kf.lam_, kf.h_], order="F")
            xyz_nco = frames.lla2ecef(lla_nco)
            xyzv_nco = frames.ned2ecefv(np.array([kf.vn_, kf.ve_, kf.vd_], order="F"), lla_nco)
            next_tT = nco_state["tT"] + conf["meas_dt"]
            obs_sim.GetRangeAndRate(
                next_tT,
                xyz_nco,
                xyzv_nco,
                kf.cb_,
                kf.cd_,
                u,
                nco_state["sv_clk"],
                nco_state["sv_pos"],
                nco_state["sv_vel"],
                rng,
                nco_state["psr"],
                nco_state["psrdot"],
                empty2,
            )

            # update nco
            next_tR = next_tT + nco_state["psr"] / LIGHT_SPEED  # -nco_state["sv_clk"][0, :]
            nco_state["chiprate"] = (GPS_CA_CODE_RATE * conf["meas_dt"] - 0.0) / (
                next_tR - nco_state["tR"]
            )
            nco_state["omega"] = TWO_PI * (conf["intmd_freq"] - nco_state["psrdot"] / lamb)

            k_meas = 0

        k_meas += 1
        k_corr += 1
        # print("---------------------------------------------------------------------")

    # close files
    nav_file.close()
    err_file.close()
    var_file.close()
    [chn_files[i].close() for i in range(M)]

    return


def vt_array(
    conf: dict,
    eph: np.ndarray[ephemeris.KeplerElements],
    atm: np.ndarray[atmosphere.KlobucharElements],
    truth: pd.DataFrame,
    seed: int = 1,
    run_num: int = 1,
) -> None:
    """
    VT
    ==

    Simulates the correlators and NCOs of a GPS L1 C/A vector tracking receiver. It makes use of the
    common transmit time (CTT) such that the measurements from each satellite can be utilized
    synchronously. This significantly simplifies the simulation by allowing the satellite transmit
    times (tT) to all be synchronized to the time of week (ToW) but allows the NCOs to adjust the
    perceived receive times (tR) of the signals.

    Inputs
    ------
    conf : dict
        Contains the configuration parameters for the simulation
            - scenario : str
            - data_file : str
            - ephem_file : str
            - out_folder : str
            - n_runs : int
            - clock_model : str
            - vel_process_std : float
            - att_process_std : float
            - sec_to_skip : float
            - init_tow : float
            - intmd_freq : float
            - sim_dt : float
            - meas_dt : float
            - tap_epl : float
            - init_cb : float
            - init_cd : float
            - init_cov : list
            - add_init_err : bool
            - jammer_modulation : str
            - jammer_type : str
            - j2s : float
            - is_multi_antenna : bool
            - n_ant : int
            - ant_xyz_0 : float
            - ant_xyz_1 : float
            - ant_xyz_2 : float
            - ant_xyz_3 : float

    eph : pd.DataFrame
        Contains the satellite ephemeris for each satellite
            - iode : float
            - iodc : float
            - toe : float
            - toc : float
            - tgd : float
            - af2 : float
            - af1 : float
            - af0 : float
            - e : float
            - sqrtA : float
            - deltan : float
            - m0 : float
            - omega0 : float
            - omega : float
            - omegaDot : float
            - i0 : float
            - iDot : float
            - cuc : float
            - cus : float
            - cic : float
            - cis : float
            - crc : float
            - crs : float
            - ura : float
            - health : float

    truth : pd.DataFrame
        Contains the truth trajectory information
            t : float
            lat : float
            lon : float
            h : float
            vn : float
            ve : float
            vd : float
            r : float
            p : float
            y : float
    """

    # constants
    M = len(eph)
    lamb = LIGHT_SPEED / GPS_L1_FREQUENCY
    beta = LIGHT_SPEED / GPS_CA_CODE_RATE
    kappa = GPS_CA_CODE_RATE / GPS_L1_FREQUENCY
    r2d_sq = RAD2DEG * RAD2DEG
    J2S = 10 ** (conf["j2s"] / 10) * np.ones(M, dtype=np.double, order="F")
    k_start = int(conf["sec_to_skip"] / conf["sim_dt"])
    ant_body = []
    for i in range(conf["n_ant"]):
        ant_body.append(conf[f"ant_xyz_{i}"])
    ant_body = np.array(ant_body, order="F").T

    # noise generation
    dist = NormalDistribution(0.0, 1.0)
    eng = RandomEngine(seed)

    # receiver sensors
    clock_model = navsense.GetNavClock(conf["clock_model"])
    clk_sim = ClockSim(
        clock_model.h0,
        clock_model.h1,
        clock_model.h2,
        conf["init_cb"],
        conf["init_cd"],
        conf["sim_dt"],
        conf["add_init_err"],
        eng,
        dist,
    )
    cno_sim = CnoSim(M, lamb, GPS_CA_CODE_RATE, conf["jammer_modulation"], conf["jammer_type"])
    obs_sim = ObservableSim(eph, atm)
    corr_sim = []
    for i in range(conf["n_ant"]):
        corr_sim.append(CorrelatorSim(conf["tap_epl"], M, 2, eng, dist))
    t_sim = 0.0

    # initialize kalman filter
    init_lla = np.array(
        [truth["lat"][k_start] * DEG2RAD, truth["lon"][k_start] * DEG2RAD, truth["h"][k_start]],
        order="F",
    )
    init_vel = np.array(
        [truth["vn"][k_start], truth["ve"][k_start], truth["vd"][k_start]], order="F"
    )
    init_rpy = np.array(
        [
            truth["r"][k_start] * DEG2RAD,
            truth["p"][k_start] * DEG2RAD,
            truth["y"][k_start] * DEG2RAD,
        ],
        order="F",
    )
    init_cb = conf["init_cb"]
    init_cd = conf["init_cd"]
    if conf["add_init_err"]:
        t = 1.0 - WGS84_E2 * np.sin(init_lla[0]) ** 2
        sqt = np.sqrt(t)
        Re = WGS84_R0 / sqt
        Rn = WGS84_R0 * (1 - WGS84_E2) / (t * t / sqt)
        T_r_p = np.diag(
            [1 / (Rn + init_lla[2]), 1 / ((Re + init_lla[2]) * np.cos(init_lla[0])), -1]
        )
        sqrtP = np.diag(conf["init_P"])
        init_lla += T_r_p @ sqrtP[0:3, 0:3] @ np.random.randn(3)
        init_vel += sqrtP[3:6, 3:6] @ np.random.randn(3)
        init_vel += sqrtP[6:9, 6:9] @ np.random.randn(3)
        init_cb += sqrtP[9, 9] @ np.random.randn()
        init_cd += sqrtP[10, 10] @ np.random.randn()
    kf = KinematicNav(
        init_lla[0],
        init_lla[1],
        init_lla[2],
        init_vel[0],
        init_vel[1],
        init_vel[2],
        init_rpy[0],
        init_rpy[1],
        init_rpy[2],
        init_cb,
        init_cd,
    )
    kf.SetClockSpec(clock_model.h0, clock_model.h1, clock_model.h2)
    kf.SetProcessNoise(conf["vel_process_psd"], conf["att_process_psd"])

    # initialize tracking states
    rng = np.zeros(M, dtype=np.double, order="F")
    empty2 = np.zeros(M, dtype=np.double, order="F")
    u = np.zeros((3, M), dtype=np.double, order="F")
    true_state = []
    for i in range(conf["n_ant"]):
        true_state.append(
            {
                "psr": np.zeros(M, dtype=np.double, order="F"),
                "psrdot": np.zeros(M, dtype=np.double, order="F"),
                "chip": np.zeros(M, dtype=np.double, order="F"),
                "chiprate": np.zeros(M, dtype=np.double, order="F"),
                "phase": np.zeros(M, dtype=np.double, order="F"),
                "omega": np.zeros(M, dtype=np.double, order="F"),
                "sv_clk": np.zeros((3, M), dtype=np.double, order="F"),
                "sv_pos": np.zeros((3, M), dtype=np.double, order="F"),
                "sv_vel": np.zeros((3, M), dtype=np.double, order="F"),
                "ToW": conf["init_tow"] * np.ones(M, dtype=np.double, order="F"),
            }
        )
    nco_state = {
        "psr": np.zeros(M, dtype=np.double, order="F"),
        "psrdot": np.zeros(M, dtype=np.double, order="F"),
        "chip": np.zeros(M, dtype=np.double, order="F"),
        "chiprate": np.zeros(M, dtype=np.double, order="F"),
        "phase": np.zeros(M, dtype=np.double, order="F"),
        "omega": np.zeros(M, dtype=np.double, order="F"),
        "tT": conf["init_tow"] * np.ones(M, dtype=np.double, order="F"),
        "tR": np.zeros(M, dtype=np.double, order="F"),
        "sv_clk": np.zeros((3, M), dtype=np.double, order="F"),
        "sv_pos": np.zeros((3, M), dtype=np.double, order="F"),
        "sv_vel": np.zeros((3, M), dtype=np.double, order="F"),
    }

    # true state
    true_cb, true_cd = clk_sim.GetCurrentState()
    lla_true = np.array(
        [truth["lat"][k_start] * DEG2RAD, truth["lon"][k_start] * DEG2RAD, truth["h"][k_start]],
        order="F",
    )
    xyz_true = frames.lla2ecef(lla_true)
    rpy_true = np.array(
        [
            truth["r"][k_start] * DEG2RAD,
            truth["p"][k_start] * DEG2RAD,
            truth["y"][k_start] * DEG2RAD,
        ],
        order="F",
    )
    C_b_n_true = attitude.euler2dcm(rpy_true, True)
    C_n_e_true = frames.ned2ecefDcm(lla_true)
    xyzv_true = C_n_e_true @ np.array(
        [truth["vn"][k_start], truth["ve"][k_start], truth["vd"][k_start]], order="F"
    )
    ant_xyz_ecef = np.asfortranarray(C_n_e_true @ (C_b_n_true @ ant_body) + xyz_true[:, None])
    for i in range(conf["n_ant"]):
        obs_sim.GetRangeAndRate(
            true_state[i]["ToW"],
            ant_xyz_ecef[:, i],
            xyzv_true,
            true_cb,
            true_cd,
            u,
            true_state[i]["sv_clk"],
            true_state[i]["sv_pos"],
            true_state[i]["sv_vel"],
            rng,
            true_state[i]["psr"],
            true_state[i]["psrdot"],
            empty2,
        )
        true_state[i]["phase"] = TWO_PI * (conf["intmd_freq"] * t_sim - true_state[i]["psr"] / lamb)
        true_state[i]["omega"] = TWO_PI * (conf["intmd_freq"] - true_state[i]["psrdot"] / lamb)
        true_state[i]["chip"] = true_state[i]["psr"] / beta
        true_state[i]["chiprate"] = GPS_CA_CODE_RATE - kappa * true_state[i]["psrdot"] / lamb

    # nco state
    rpy_nco = np.zeros(3, order="F")
    lla_nco = np.array([kf.phi_, kf.lam_, kf.h_], order="F")
    xyz_nco = frames.lla2ecef(lla_nco)
    xyzv_nco = frames.ned2ecefv(np.array([kf.vn_, kf.ve_, kf.vd_], order="F"), lla_nco)
    obs_sim.GetRangeAndRate(
        nco_state["tT"],
        xyz_nco,
        xyzv_nco,
        kf.cb_,
        kf.cd_,
        u,
        nco_state["sv_clk"],
        nco_state["sv_pos"],
        nco_state["sv_vel"],
        rng,
        nco_state["psr"],
        nco_state["psrdot"],
        empty2,
    )
    nco_state["phase"] = TWO_PI * (conf["intmd_freq"] * t_sim - nco_state["psr"] / lamb)
    nco_state["omega"] = TWO_PI * (conf["intmd_freq"] - nco_state["psrdot"] / lamb)
    nco_state["chip"] = nco_state["psr"] / beta
    nco_state["chiprate"] = GPS_CA_CODE_RATE - kappa * nco_state["psrdot"] / lamb
    nco_state["tR"] = nco_state["tT"] + nco_state["psr"] / LIGHT_SPEED

    # initialize cno/discriminator estimators
    cno_state = {
        "alpha": 0.0025,
        "est_cno": 1000 * np.ones(M, dtype=np.double, order="F"),  # 25119
        "true_cno": np.zeros(M, dtype=np.double, order="F"),
        "detectors": [lockdetectors.LockDetectors(0.0025) for _ in range(M)],
    }
    dR = np.zeros(M, order="F")
    dRR = np.zeros(M, order="F")
    dP = np.zeros((conf["n_ant"], M), order="F")
    dR_var = np.zeros(M, order="F")
    dRR_var = np.zeros(M, order="F")
    dP_var = np.zeros((conf["n_ant"], M), order="F")

    # open files for binary output
    mypath = Path(conf["out_folder"]) / conf["scenario"] / str(run_num)
    mypath.mkdir(parents=True, exist_ok=True)
    nav_file = open(mypath / "Nav_Results_Log.bin", "wb")
    err_file = open(mypath / "Err_Results_Log.bin", "wb")
    var_file = open(mypath / "Var_Results_Log.bin", "wb")
    chn_files = [open(mypath / f"Channel_{i}_Results_Log.bin", "wb") for i in range(M)]

    # initialize correlators and beamsteering
    R1 = [np.zeros((3, M), dtype=np.complex128, order="F")] * conf["n_ant"]
    R2 = [np.zeros((3, M), dtype=np.complex128, order="F")] * conf["n_ant"]
    R = [np.zeros((3, M), dtype=np.complex128, order="F")] * conf["n_ant"]
    R1_BS = np.zeros((3, M), dtype=np.complex128, order="F")
    R2_BS = np.zeros((3, M), dtype=np.complex128, order="F")
    R_BS = np.zeros((3, M), dtype=np.complex128, order="F")
    U_BODY = C_b_n_true.T @ (C_n_e_true.T @ u)
    spatial_phase = TWO_PI / lamb * (ant_body.T @ U_BODY)
    W_BS = np.exp(1j * spatial_phase)

    # run simulation
    k_iq = 1
    k_meas = 1
    k_corr = 1
    dk_meas = int(conf["meas_dt"] / conf["sim_dt"])
    dk_corr = int(dk_meas / 2)
    kf.Propagate(conf["meas_dt"])
    for k_sim in range(k_start + 1, len(truth)):
        true_cb, true_cd = clk_sim.Simulate()
        t_sim += conf["sim_dt"]

        # propagate truth states
        lla_true = np.array(
            [truth["lat"][k_sim] * DEG2RAD, truth["lon"][k_sim] * DEG2RAD, truth["h"][k_sim]],
            order="F",
        )
        rpy_true = np.array(
            [
                truth["r"][k_sim] * DEG2RAD,
                truth["p"][k_sim] * DEG2RAD,
                truth["y"][k_sim] * DEG2RAD,
            ],
            order="F",
        )
        attitude.euler2dcm(C_b_n_true, rpy_true, True)
        frames.ned2ecefDcm(C_n_e_true, lla_true)
        frames.lla2ecef(xyz_true, lla_true)
        xyzv_true = C_n_e_true @ np.array(
            [truth["vn"][k_sim], truth["ve"][k_sim], truth["vd"][k_sim]], order="F"
        )
        ant_xyz_ecef = np.asfortranarray(C_n_e_true @ (C_b_n_true @ ant_body) + xyz_true[:, None])
        for i in range(conf["n_ant"]):
            true_state[i]["ToW"] += conf["sim_dt"]
            old_chip = true_state[i]["chip"]

            obs_sim.GetRangeAndRate(
                true_state[i]["ToW"],
                ant_xyz_ecef[:, i],
                xyzv_true,
                true_cb,
                true_cd,
                u,
                true_state[i]["sv_clk"],
                true_state[i]["sv_pos"],
                true_state[i]["sv_vel"],
                rng,
                true_state[i]["psr"],
                true_state[i]["psrdot"],
                empty2,
            )
            true_state[i]["phase"] = TWO_PI * (
                conf["intmd_freq"] * t_sim - true_state[i]["psr"] / lamb
            )
            true_state[i]["omega"] = TWO_PI * (conf["intmd_freq"] - true_state[i]["psrdot"] / lamb)
            true_state[i]["chip"] = true_state[i]["psr"] / beta
            true_state[i]["chiprate"] = (
                GPS_CA_CODE_RATE - (true_state[i]["chip"] - old_chip) / conf["sim_dt"]
            )

        # propagate nco states
        t_int = (GPS_CA_CODE_RATE / nco_state["chiprate"]) * conf["sim_dt"]
        nco_state["tT"] += conf["sim_dt"]
        nco_state["tR"] += t_int
        nco_state["chip"] = (nco_state["tR"] - nco_state["tT"]) * GPS_CA_CODE_RATE
        nco_state["phase"] += nco_state["omega"] * t_int

        # get true cno
        cno_state["true_cno"] = cno_sim.FsplPlusJammerModel(J2S, rng)

        # update correlators
        for i in range(conf["n_ant"]):
            corr_sim[i].NextSample(
                conf["sim_dt"],
                cno_state["true_cno"],
                true_state[i]["chip"],
                true_state[i]["chiprate"],
                true_state[i]["phase"],
                true_state[i]["omega"],
                nco_state["chip"],
                nco_state["chiprate"],
                nco_state["phase"],
                nco_state["omega"],
            )

        # extract correlators
        if k_corr == dk_corr:
            if k_iq == 1:
                for i in range(conf["n_ant"]):
                    R1[i] = corr_sim[i].GetCorrelators()
                k_iq = 2
            else:
                for i in range(conf["n_ant"]):
                    R2[i] = corr_sim[i].GetCorrelators()
                k_iq = 1
            k_corr = 0

        # measurement update
        if k_meas == dk_meas:
            # complete integration and beam steer (post correlation)
            R_BS[:] = 0.0
            R1_BS[:] = 0.0
            R2_BS[:] = 0.0
            for i in range(conf["n_ant"]):
                R[i] = R1[i] + R2[i]
                for j in range(M):
                    R_BS[:, j] += W_BS[i, j] * R[i][:, j]
                    R1_BS[:, j] += W_BS[i, j] * R1[i][:, j]
                    R2_BS[:, j] += W_BS[i, j] * R2[i][:, j]
            # R_BS = R[0]
            # R1_BS = R1[0]
            # R2_BS = R2[0]

            # update cno/discriminators
            for j in range(M):
                cno_state["detectors"][j].Update(R_BS[1, j], conf["meas_dt"])
                # cno_state["detectors"][j].Update(R[0][1, j], conf["meas_dt"])
                cno_state["est_cno"][j] = cno_state["detectors"][j].GetCno()
                dR[j] = beta * discriminator.DllNneml(R_BS[0, j], R_BS[2, j])
                dRR[j] = (
                    -lamb
                    / TWO_PI
                    * discriminator.FllAtan2(R1_BS[1, j], R2_BS[1, j], conf["meas_dt"])
                )
                dR_var[j] = beta**2 * discriminator.DllVariance(
                    cno_state["est_cno"][j], conf["meas_dt"]
                )
                dRR_var[j] = (lamb / TWO_PI) ** 2 * discriminator.FllVariance(
                    cno_state["est_cno"][j], conf["meas_dt"]
                )
                for i in range(conf["n_ant"]):
                    dP[i, j] = discriminator.PllAtan2(R[i][1, j])
                    # dP[i, j] = discriminator.PllAtan2(R_BS[1, j])
                    dP_var[i, j] = 2.0 * discriminator.PllVariance(
                        cno_state["true_cno"][j], conf["meas_dt"]
                    )
            # U_NED = C_n_e_true.T @ u
            # dP = -TWO_PI / lamb * ((C_b_n_true @ ant_body).T @ U_NED)
            # dP = np.fmod((-dP + dP[0, :]) + PI, TWO_PI) - PI
            dP = np.fmod((dP - dP[0, :]) + PI, TWO_PI) - PI
            dP[dP > PI] -= TWO_PI
            dP[dP < -PI] += TWO_PI

            # get satellite and measurement information
            lla_nco = np.array([kf.phi_, kf.lam_, kf.h_], order="F")
            frames.lla2ecef(xyz_nco, lla_nco)
            frames.ned2ecefv(xyzv_nco, np.array([kf.vn_, kf.ve_, kf.vd_], order="F"), lla_nco)
            obs_sim.GetRangeAndRate(
                nco_state["tT"],
                xyz_nco,
                xyzv_nco,
                kf.cb_,
                kf.cd_,
                u,
                nco_state["sv_clk"],
                nco_state["sv_pos"],
                nco_state["sv_vel"],
                rng,
                nco_state["psr"],
                nco_state["psrdot"],
                empty2,
            )

            # kalman update
            psr_meas = (nco_state["tR"] - nco_state["tT"]) * LIGHT_SPEED + dR
            psrdot_meas = -lamb * (nco_state["omega"] / TWO_PI - conf["intmd_freq"]) + dRR
            # kf.GnssUpdate(
            #     nco_state["sv_pos"], nco_state["sv_vel"], psr_meas, psrdot_meas, dR_var, dRR_var
            # )
            kf.PhasedArrayUpdate(
                nco_state["sv_pos"],
                nco_state["sv_vel"],
                psr_meas,
                psrdot_meas,
                dP,
                dR_var,
                dRR_var,
                dP_var,
                ant_body,
                conf["n_ant"],
                lamb / TWO_PI,
            )

            # save to binary files
            attitude.dcm2euler(rpy_nco, kf.C_b_l_, True)
            rpy_nco *= RAD2DEG
            lla_nco = np.array([kf.phi_, kf.lam_, kf.h_], order="F")
            data = [
                truth["t"][k_sim],
                nco_state["tT"][0],
                kf.phi_ * RAD2DEG,
                kf.lam_ * RAD2DEG,
                kf.h_,
                kf.vn_,
                kf.ve_,
                kf.vd_,
                rpy_nco[0],
                rpy_nco[1],
                rpy_nco[2],
                kf.cb_,
                kf.cd_,
            ]
            nav_file.write(pack("d" * 13, *data))
            ned_err = frames.lla2ned(lla_nco, lla_true)
            rpy_err = attitude.dcm2euler(C_b_n_true @ kf.C_b_l_.T, True) * RAD2DEG
            data = [
                truth["t"][k_sim],
                nco_state["tT"][0],
                ned_err[0],
                ned_err[1],
                ned_err[2],
                truth["vn"][k_sim] - kf.vn_,
                truth["ve"][k_sim] - kf.ve_,
                truth["vd"][k_sim] - kf.vd_,
                rpy_err[0],
                rpy_err[1],
                rpy_err[2],
                true_cb - kf.cb_,
                true_cd - kf.cd_,
            ]
            err_file.write(pack("d" * 13, *data))
            data = [
                truth["t"][k_sim],
                nco_state["tT"][0],
                kf.P_[0, 0],
                kf.P_[1, 1],
                kf.P_[2, 2],
                kf.P_[3, 3],
                kf.P_[4, 4],
                kf.P_[5, 5],
                kf.P_[6, 6] * r2d_sq,
                kf.P_[7, 7] * r2d_sq,
                kf.P_[8, 8] * r2d_sq,
                kf.P_[9, 9],
                kf.P_[10, 10],
            ]
            var_file.write(pack("d" * 13, *data))
            C_e_n_nco = frames.ecef2nedDcm(lla_nco)
            U_NED = C_e_n_nco @ u
            if conf["n_ant"] == 4:
                P_reg = np.array([R[0][1, :], R[1][1, :], R[2][1, :], R[3][1, :]])
            elif conf["n_ant"] == 3:
                P_reg = np.array([R[0][1, :], R[1][1, :], R[2][1, :], np.zeros(M)])
            elif conf["n_ant"] == 2:
                P_reg = np.array([R[0][1, :], R[1][1, :], np.zeros(M), np.zeros(M)])
            for i in range(M):
                data = [
                    truth["t"][k_sim],
                    true_state[0]["ToW"][i],
                    180 + RAD2DEG * np.atan2(U_NED[1, i], U_NED[0, i]),
                    RAD2DEG * np.asin(U_NED[2, i]),  # these angles should be ENU not NED
                    true_state[0]["phase"][i],
                    true_state[0]["omega"][i],
                    true_state[0]["chip"][i],
                    true_state[0]["chiprate"][i],
                    10.0 * np.log10(cno_state["true_cno"][i]),
                    nco_state["phase"][i],
                    nco_state["omega"][i],
                    nco_state["chip"][i],
                    nco_state["chiprate"][i],
                    10.0 * np.log10(cno_state["est_cno"][i]),
                    R_BS[0, i].real,
                    R_BS[0, i].imag,
                    R_BS[1, i].real,
                    R_BS[1, i].imag,
                    R_BS[2, i].real,
                    R_BS[2, i].imag,
                    R1_BS[1, i].real,
                    R1_BS[1, i].imag,
                    R2_BS[1, i].real,
                    R2_BS[1, i].imag,
                    P_reg[0, i].real,
                    P_reg[0, i].imag,
                    P_reg[1, i].real,
                    P_reg[1, i].imag,
                    P_reg[2, i].real,
                    P_reg[2, i].imag,
                    P_reg[3, i].real,
                    P_reg[3, i].imag,
                ]
                chn_files[i].write(pack("d" * 32, *data))

            # propagate
            kf.Propagate(conf["meas_dt"])
            lla_nco = np.array([kf.phi_, kf.lam_, kf.h_], order="F")
            frames.lla2ecef(xyz_nco, lla_nco)
            frames.ned2ecefv(xyzv_nco, np.array([kf.vn_, kf.ve_, kf.vd_], order="F"), lla_nco)
            next_tT = nco_state["tT"] + conf["meas_dt"]
            obs_sim.GetRangeAndRate(
                next_tT,
                xyz_nco,
                xyzv_nco,
                kf.cb_,
                kf.cd_,
                u,
                nco_state["sv_clk"],
                nco_state["sv_pos"],
                nco_state["sv_vel"],
                rng,
                nco_state["psr"],
                nco_state["psrdot"],
                empty2,
            )

            # update nco
            next_tR = next_tT + nco_state["psr"] / LIGHT_SPEED  # -nco_state["sv_clk"][0, :]
            nco_state["chiprate"] = (GPS_CA_CODE_RATE * conf["meas_dt"] - 0.0) / (
                next_tR - nco_state["tR"]
            )
            nco_state["omega"] = TWO_PI * (conf["intmd_freq"] - nco_state["psrdot"] / lamb)

            # update beam steering weights
            U_BODY = kf.C_b_l_.T @ (C_e_n_nco @ u)
            # U_BODY = C_b_n_true.T @ (C_n_e_true.T @ u)
            spatial_phase = TWO_PI / lamb * (ant_body.T @ U_BODY)
            W_BS = np.exp(1j * spatial_phase)

            k_meas = 0

        k_meas += 1
        k_corr += 1

    # close files
    nav_file.close()
    err_file.close()
    var_file.close()
    [chn_files[i].close() for i in range(M)]

    return
