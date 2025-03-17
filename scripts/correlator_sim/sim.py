from parsers import ParseConfig, ParseEphem, ParseNavStates
import numpy as np
import pandas as pd
from pathlib import Path
import struct
from navtools import (
    TWO_PI,
    LIGHT_SPEED,
    RAD2DEG,
    DEG2RAD,
    WGS84_E2,
    WGS84_R0,
    frames,
    math,
    attitude,
)
from satutils import GPS_L1_FREQUENCY, GPS_CA_CODE_RATE, GPS_CA_CODE_LENGTH, ephemeris, atmosphere
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

    # initialize cno estimators
    cno_state = {
        "alpha": 0.0025,
        "est_cno": np.zeros(M, dtype=np.double, order="F"),
        "true_cno": np.zeros(M, dtype=np.double, order="F"),
        "detectors": [lockdetectors.LockDetectors(0.0025) for _ in range(M)],
    }

    # open files for binary output
    mypath = Path(conf["out_folder"]) / conf["scenario"] / str(run_num)
    mypath.mkdir(parents=True, exist_ok=True)
    nav_file = open(mypath / "Nav_Results_Log.bin", "wb")
    err_file = open(mypath / "Err_Results_Log.bin", "wb")
    var_file = open(mypath / "Var_Results_Log.bin", "wb")
    chn_files = [open(mypath / f"Channel_{i}_Results_Log.bin", "wb") for i in range(M)]

    # run simulation
    true_cb = 0.0
    true_cd = 0.0
    k_iq = 1
    k_meas = 1
    k_corr = 1
    dk_meas = int(conf["meas_dt"] / conf["sim_dt"])
    dk_corr = dk_meas / 2
    kf.Propagate(conf["meas_dt"])
    for k_sim in range(k_start, len(truth)):
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
        nco_state["phase"] = nco_state["phase"] + nco_state["omega"] * t_int

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
            dR = np.zeros(M, order="F")
            dRR = np.zeros(M, order="F")
            dR_var = np.zeros(M, order="F")
            dRR_var = np.zeros(M, order="F")
            for i in range(M):
                cno_state["detectors"][i].Update(R[1, i], conf["meas_dt"])
                cno_state["est_cno"][i] = cno_state["detectors"][i].GetCno()
                dR[i] = beta * discriminator.DllNneml2(R[0, i], R[2, i])
                dRR[i] = -lamb * discriminator.FllAtan2(R1[1, i], R2[1, i], conf["meas_dt"])
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
            psr_meas = (nco_state["tR"] - nco_state["tT"]) * LIGHT_SPEED + dR
            psrdot_meas = -lamb * (nco_state["omega"] / TWO_PI - conf["intmd_freq"]) + dRR

            # kalman update
            kf.GnssUpdate(
                nco_state["sv_pos"], nco_state["sv_vel"], psr_meas, psrdot_meas, dR_var, dRR_var
            )

            # save to binary files
            rpy_nco = attitude.dcm2euler(kf.C_b_l_) * RAD2DEG
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
            nav_file.write(struct.pack("d" * 13, *data))
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
                truth["r"][k_sim] - rpy_nco[0],
                truth["p"][k_sim] - rpy_nco[1],
                truth["y"][k_sim] - rpy_nco[2],
                true_cb - kf.cb_,
                true_cd - kf.cd_,
            ]
            err_file.write(struct.pack("d" * 13, *data))
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
            var_file.write(struct.pack("d" * 13, *data))
            for i in range(M):
                data = [
                    truth["t"][k_sim],
                    true_state["ToW"][i],
                    true_state["phase"][i],
                    true_state["omega"][i],
                    true_state["chip"][i],
                    true_state["chiprate"][i],
                    cno_state["true_cno"][i],
                    nco_state["phase"][i],
                    nco_state["omega"][i],
                    nco_state["chip"][i],
                    nco_state["chiprate"][i],
                    cno_state["est_cno"][i],
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
                    0.0,
                    dRR[i],
                    dR[i],
                ]
                chn_files[i].write(struct.pack("d" * 25, *data))

            # propagate
            kf.Propagate(conf["meas_dt"])
            lla_nco = np.array([kf.phi_, kf.lam_, kf.h_], order="F")
            xyz_nco = frames.lla2ecef(lla_nco)
            xyzv_nco = frames.ned2ecefv(np.array([kf.vn_, kf.ve_, kf.vd_], order="F"), lla_nco)
            obs_sim.GetRangeAndRate(
                nco_state["tT"] + conf["meas_dt"],
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
            next_tR = nco_state["tT"] + conf["meas_dt"] + nco_state["psr"] / LIGHT_SPEED
            nco_state["chiprate"] = (GPS_CA_CODE_RATE * conf["meas_dt"] - 0.0) / (
                next_tR - nco_state["tR"]
            )
            nco_state["omega"] = TWO_PI * (conf["intmd_freq"] - nco_state["psrdot"] / lamb)

            k_meas = 0

        k_meas += 1
        k_corr += 1

    # close files
    nav_file.close()
    err_file.close()
    var_file.close()
    [chn_files[i].close() for i in range(M)]

    return


if __name__ == "__main__":
    conf = ParseConfig("config/vt_correlator_sim.yaml")
    eph, atm = ParseEphem("data/sim_ephem.bin")
    truth = ParseNavStates("data/sim_truth.bin")
    vt(conf, eph, atm, truth)
