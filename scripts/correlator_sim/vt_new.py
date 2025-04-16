import sys
import numpy as np
from io import BufferedWriter
from struct import pack
from pathlib import Path
from scipy.interpolate import make_splrep, BSpline
from dataclasses import dataclass
from models import ClockModel, CnoModel, CorrelatorModel, ObservableModel
from navtools import DEG2RAD, RAD2DEG, PI, TWO_PI, LIGHT_SPEED
from navtools._navtools_core.attitude import euler2quat, quat2euler
from navtools._navtools_core.frames import lla2ecef, lla2ned, ned2ecefv, ecef2nedDcm
from navtools._navtools_core.math import quatdot
from satutils import GPS_CA_CODE_RATE, GPS_L1_FREQUENCY, GPS_CA_CODE_LENGTH
from sturdins import KinematicNav
from sturdr._sturdr_core.lockdetectors import LockDetectors
from sturdr._sturdr_core.discriminator import (
    DllNneml2,
    DllVariance,
    FllAtan2,
    FllVariance,
    PllAtan2,
    PllVariance,
)

sys.path.append("scripts")
from utils.parsers import ParseConfig, ParseEphem, ParseNavSimStates

M2NS = 1e9 / LIGHT_SPEED
M2NS_SQ = M2NS * M2NS
R2D_SQ = RAD2DEG * RAD2DEG


@dataclass(slots=True)
class TruthObservables:
    t: BSpline
    lat: BSpline
    lon: BSpline
    h: BSpline
    vn: BSpline
    ve: BSpline
    vd: BSpline
    r: BSpline
    p: BSpline
    y: BSpline
    cb: BSpline
    cd: BSpline
    tR: BSpline
    tT: list[BSpline]
    cno: list[BSpline]
    psr: list[BSpline]
    psrdot: list[BSpline]


@dataclass(slots=True)
class NcoState:
    ToW: np.double  # same a tT but modded to 0.02
    tT: np.double
    chip: np.double
    chiprate: np.double
    phase: np.double
    omega: np.double
    current_sample: int
    total_sample: int
    half_sample: int
    model: CorrelatorModel
    obs: ObservableModel
    locks: LockDetectors
    unit_vec: np.ndarray[np.double, 3]
    E: np.complex128
    P: np.complex128
    L: np.complex128
    P1: np.complex128
    P2: np.complex128


class VectorTrackingSim:
    _conf: dict
    _gen: np.random.Generator
    _correlation_scheme: callable
    _discriminator_scheme: callable
    _vector_update: callable
    _clock_model: ClockModel
    _cno_model: CnoModel
    _channels: list[NcoState]
    _truth: TruthObservables
    _M: int
    _tR: np.double
    _T_end: np.double
    _T_ms: int
    _lambda: np.double
    _beta: np.double
    _kappa: np.double
    _intmd_freq: np.double
    _channel_order: np.ndarray[int]
    _delta_samp: np.ndarray[int]
    _nav_file: BufferedWriter
    _err_file: BufferedWriter
    _chn_files: list[BufferedWriter]

    def __init__(self, file: str | Path, seed: int = None, run_index: int = 1):
        # parse configuration
        self._conf = ParseConfig(file)
        self._T_ms = int(self._conf["meas_dt"] * 1000)
        self._lambda = LIGHT_SPEED / (TWO_PI * GPS_L1_FREQUENCY)
        self._beta = LIGHT_SPEED / GPS_CA_CODE_RATE
        self._kappa = GPS_CA_CODE_RATE / (TWO_PI * GPS_L1_FREQUENCY)
        self._intmd_freq = TWO_PI * self._conf["intmd_freq"]

        # random number generator
        self._gen = np.random.default_rng(seed)

        # generate truth observables
        self.__init_truth_states()

        # initialize nco tracking states
        self.__init_nco()

        # initialize kalman filter
        self._kf = KinematicNav(
            self._truth.lat(self._tR),
            self._truth.lon(self._tR),
            self._truth.h(self._tR),
            self._truth.vn(self._tR),
            self._truth.ve(self._tR),
            self._truth.vd(self._tR),
            self._truth.r(self._tR),
            self._truth.p(self._tR),
            self._truth.y(self._tR),
            self._truth.cb(self._tR) * LIGHT_SPEED,
            self._truth.cd(self._tR) * LIGHT_SPEED,
        )
        self._kf.SetClockSpec(
            self._clock_model._model.h0, self._clock_model._model.h1, self._clock_model._model.h2
        )
        self._kf.SetProcessNoise(self._conf["vel_process_psd"], self._conf["att_process_psd"])

        # open files for binary output
        mypath = Path(self._conf["out_folder"]) / self._conf["scenario"] / f"Run{run_index}"
        mypath.mkdir(parents=True, exist_ok=True)
        self._nav_file = open(mypath / "Nav_Results_Log.bin", "wb")
        self._err_file = open(mypath / "Err_Results_Log.bin", "wb")
        self._chn_files = [
            open(mypath / f"Channel_{i}_Results_Log.bin", "wb") for i in range(self._M)
        ]

        # apply correct functionality based on antenna configuration
        return

    def __del__(self):
        self._nav_file.close()
        self._err_file.close()
        for f in self._chn_files:
            f.close()
        return

    def Run(self):
        """
        Simulates an asynchronous vector tracking receiver
        """
        # count = 0
        next_tR = self._tR + self._conf["meas_dt"]
        while next_tR < self._T_end:
            # samp_processed = 0
            # samp_per_20ms = int(self._conf["meas_dt"] * self._conf["samp_freq"])

            # run an update from each channel
            for ii in self._channel_order:
                # determine tR at beginning, middle, and end of correlation period
                dt1 = self._channels[ii].half_sample / self._conf["samp_freq"]
                dt2 = (
                    self._channels[ii].total_sample - self._channels[ii].half_sample
                ) / self._conf["samp_freq"]
                t0 = self._tR - self._channels[ii].current_sample / self._conf["samp_freq"]
                t1 = t0 + dt1
                t2 = t1 + dt2
                # print(f"t0 = {t0}")
                # print(f"t1 = {t1}")
                # print(f"t2 = {t2}")

                # correlate first and second halves
                R1 = self.__correlate(ii, t1, dt1)
                R2 = self.__correlate(ii, t2, dt2)

                # combine correlators
                R = R1 + R2
                self._channels[ii].E, self._channels[ii].P, self._channels[ii].L = R
                self._channels[ii].P1 = R1[1]
                self._channels[ii].P2 = R2[1]

                # run lock detectors
                self._channels[ii].locks.Update(self._channels[ii].P, self._conf["meas_dt"])

                # update channel states
                self._channels[ii].ToW += self._conf["meas_dt"]
                self._channels[ii].ToW = np.round(self._channels[ii].ToW, 2)
                self._channels[ii].chip -= self._T_ms * GPS_CA_CODE_LENGTH

                # vector update
                d_samp = self._channels[ii].total_sample - self._channels[ii].current_sample
                self.VectorUpdate(self._channels[ii], d_samp)

                # move all other channels "current_sample" forward
                # print(
                #     f"{d_samp:6d}, {self._channels[ii].E:.2f}, {self._channels[ii].P:.2f}, "
                #     f"{self._channels[ii].L:.2f}, {10*np.log10(self._channels[ii].locks.GetCno()):.2f}"
                # )
                # samp_processed += d_samp
                for jj in range(self._M):
                    self._channels[jj].current_sample += d_samp
                self._channels[ii].current_sample = 0

                # use vector nco frequencies to predict next update
                self._channels[ii].total_sample, self._channels[ii].half_sample = (
                    self.NewCodePeriod(self._channels[ii].chiprate, self._channels[ii].chip)
                )

                # increment
                next_tR = self._tR + self._conf["meas_dt"]

            # log results to binary file
            # count += 1
            # samp_remaining = (
            #     self._channels[self._channel_order[0]].total_sample
            #     - self._channels[self._channel_order[0]].current_sample
            # )
            # print("---------------------------------------------------------------------------")
            # print(count)
            # print(f"Receive Time = {self._tR}")
            # print(f"Samples processed this period = {samp_processed} / {samp_per_20ms}")
            # print(f"Samples remaining = {samp_remaining}")
            self.__update_processing_order()
            self.__log_to_file()
        return

    def NewCodePeriod(self, chiprate: np.double, rem_code_phase: np.double) -> tuple[int, int]:
        """
        Predicts the number of samples required to complete the next code period
        """
        code_phase_step = chiprate / self._conf["samp_freq"]
        total_samp = int((self._T_ms * GPS_CA_CODE_LENGTH - rem_code_phase) / code_phase_step)
        half_samp = total_samp // 2
        return total_samp, half_samp

    def Discriminators(self, channel: NcoState) -> tuple[float, float, float, float]:
        """
        Calculates the code and frequency tracking errors
        """
        cno = channel.locks.GetCno()
        psr_err = self._beta * DllNneml2(channel.E, channel.L)
        psr_var = self._beta**2 * DllVariance(cno, self._conf["meas_dt"])
        psrdot_err = self._lambda * FllAtan2(channel.P1, channel.P2, self._conf["meas_dt"])
        psrdot_var = self._lambda**2 * FllVariance(cno, self._conf["meas_dt"])
        # print(f"psr_err = {psr_err} | psrdot_err = {psrdot_err}")
        return psr_err, psr_var, psrdot_err, psrdot_var

    def VectorUpdate(self, channel: NcoState, delta_samp: int) -> None:
        """
        Runs a traditional Vector Delay-Frequency Lock Loop measurement update
        """
        # 1. Update satellite pos, vel, and clock terms from transmit time
        _tT = (
            channel.ToW
            + channel.chip / channel.chiprate
            + channel.obs.GroupDelay
            - channel.obs.SatClock[0]
        )
        channel.obs.UpdateSatState(_tT)
        # _tT -= channel.obs.SatClock[0]
        _tT = (
            channel.ToW
            + channel.chip / channel.chiprate
            + channel.obs.GroupDelay
            - channel.obs.SatClock[0]
        )

        # 2. Propagate KF and receive forward by delta samples accumulated
        if delta_samp > 0:
            _dt = delta_samp / self._conf["samp_freq"]
            _tR = self._tR + _dt
            self._kf.Propagate(_dt)
        else:
            _tR = self._tR

        # 3. Estimate vector tracking residuals
        psr_err, psr_var, psrdot_err, psrdot_var = self.Discriminators(channel)

        # 4. Estimate tracking psr and psrdot (combine vector and scalar)
        _psr = np.array([LIGHT_SPEED * (_tR - _tT) - psr_err], order="F")
        _psrdot = np.array(
            [-self._lambda * channel.omega + LIGHT_SPEED * channel.obs.SatClock[1] - psrdot_err],
            order="F",
        )
        _psr_var = np.array([psr_var], order="F")
        _psrdot_var = np.array([psrdot_var], order="F")

        # 5. Call vector processing update
        self._kf.GnssUpdate(
            channel.obs.SatPos, channel.obs.SatVel, _psr, _psrdot, _psr_var, _psrdot_var
        )
        # print(
        #     f"True|Nav|Diff Bias = {self._truth.cb(_tR) * LIGHT_SPEED:.3f} | {self._kf.cb_:.3f} | "
        #     f"{(self._truth.cb(_tR) * LIGHT_SPEED - self._kf.cb_):.3f}"
        # )

        # 6. Predict ECEF state at end of next code period
        _lla = np.array([self._kf.phi_, self._kf.lam_, self._kf.h_], order="F")
        _nedv = np.array([self._kf.vn_, self._kf.ve_, self._kf.vd_], order="F")
        _vel_pred = ned2ecefv(_nedv, _lla)
        _pos_pred = lla2ecef(_lla) + _vel_pred * self._conf["meas_dt"]
        _cd_pred = self._kf.cd_
        _cb_pred = self._kf.cb_ + (_cd_pred - 0.87e-9) * self._conf["meas_dt"]
        _tT_pred = (
            channel.ToW + self._conf["meas_dt"] + channel.obs.GroupDelay - channel.obs.SatClock[0]
        )
        channel.obs.UpdateSatState(_tT_pred)
        channel.obs.CalcRangeAndRate(
            _pos_pred, _vel_pred, _cb_pred / LIGHT_SPEED, _cd_pred / LIGHT_SPEED, True
        )
        _tR_pred = channel.ToW + self._conf["meas_dt"] + channel.obs.Pseudorange / LIGHT_SPEED
        # _tR_pred = _tT_pred - channel.obs.SatClock[0] + (channel.obs.Range + _cb_pred) / LIGHT_SPEED

        # 7. Vector NCO update
        channel.chiprate = (GPS_CA_CODE_RATE * self._conf["meas_dt"] - channel.chip) / (
            _tR_pred - _tR
        )
        channel.omega = -(channel.obs.RangeRate + _cd_pred) / self._lambda
        channel.unit_vec = channel.obs.EcefUnitVec
        self._tR = _tR
        return

    def __correlate(self, ii: int, T: float, dt: float) -> None:
        """
        Correlates channel 'ii' to the true signal
        """
        omega = -self._truth.psrdot[ii](T) / self._lambda
        phase = -self._truth.psr[ii](T) / self._lambda
        chiprate = GPS_CA_CODE_RATE - (
            self._truth.psr[ii](T + 0.005) - self._truth.psr[ii](T - 0.005)
        ) / (self._beta * 0.01)
        chip = np.mod(self._truth.tT[ii](T), 0.02) * 1000 * GPS_CA_CODE_LENGTH
        self._channels[ii].phase += self._channels[ii].omega * dt
        self._channels[ii].chip += self._channels[ii].chiprate * dt
        self._channels[ii].model.NextSample(
            dt,
            self._truth.cno[ii](T),
            chip,
            chiprate,
            phase,
            omega,
            self._channels[ii].chip,
            self._channels[ii].chiprate,
            self._channels[ii].phase,
            self._channels[ii].omega,
        )
        R = self._channels[ii].model.Extract()
        return R

    def __update_processing_order(self) -> None:
        """
        Re-orders the processing of the channels based on the delta-samples remaining inside their
        individual code periods
        """
        for ii in range(self._M):
            self._delta_samp[ii] = (
                self._channels[ii].total_sample - self._channels[ii].current_sample
            )
        self._channel_order = self._delta_samp.argsort()
        # print(f"Channel order = {self._channel_order}")
        return

    def __log_to_file(self) -> None:
        """
        Writes the current states of the navigator and the channels to binary files
        """
        # calculate errors
        lla_nav = np.array([self._kf.phi_, self._kf.lam_, self._kf.h_], order="F")
        lla_true = np.array(
            [self._truth.lat(self._tR), self._truth.lon(self._tR), self._truth.h(self._tR)],
            order="F",
        )
        ned_err = lla2ned(lla_nav, lla_true)
        rpy_true = np.array(
            [self._truth.r(self._tR), self._truth.p(self._tR), self._truth.y(self._tR)], order="F"
        )
        q_true = euler2quat(rpy_true)
        q_nav = np.array(
            [self._kf.q_b_l_[0], -self._kf.q_b_l_[1], -self._kf.q_b_l_[2], -self._kf.q_b_l_[3]],
            order="F",
        )  # inverted
        q_err = quatdot(q_true, q_nav)
        rpy_err = quat2euler(q_err, True) * RAD2DEG

        # save to binary files
        data = [
            self._truth.t(self._tR),
            self._tR,
            self._kf.phi_ * RAD2DEG,
            self._kf.lam_ * RAD2DEG,
            self._kf.h_,
            self._kf.vn_,
            self._kf.ve_,
            self._kf.vd_,
            self._kf.q_b_l_[0],
            self._kf.q_b_l_[1],
            self._kf.q_b_l_[2],
            self._kf.q_b_l_[3],
            self._kf.cb_ * M2NS,  # nanoseconds
            self._kf.cd_ * M2NS,  # nanoseconds
            self._kf.P_[0, 0],
            self._kf.P_[1, 1],
            self._kf.P_[2, 2],
            self._kf.P_[3, 3],
            self._kf.P_[4, 4],
            self._kf.P_[5, 5],
            self._kf.P_[6, 6] * R2D_SQ,  # deg^2
            self._kf.P_[7, 7] * R2D_SQ,  # deg^2
            self._kf.P_[8, 8] * R2D_SQ,  # deg^2
            self._kf.P_[9, 9] * M2NS_SQ,  # nanoseconds^2
            self._kf.P_[10, 10] * M2NS_SQ,  # nanoseconds^2
        ]
        self._nav_file.write(pack("d" * 25, *data))
        data = [
            self._truth.t(self._tR),
            self._tR,
            ned_err[0],
            ned_err[1],
            ned_err[2],
            self._truth.vn(self._tR) - self._kf.vn_,
            self._truth.ve(self._tR) - self._kf.ve_,
            self._truth.vd(self._tR) - self._kf.vd_,
            rpy_err[0],
            rpy_err[1],
            rpy_err[2],
            (self._truth.cb(self._tR) * LIGHT_SPEED - self._kf.cb_) * M2NS,  # nanoseconds
            (self._truth.cd(self._tR) * LIGHT_SPEED - self._kf.cd_) * M2NS,  # nanoseconds
        ]
        self._err_file.write(pack("d" * 13, *data))

        C_e_n = ecef2nedDcm(lla_nav)
        for ii in range(self._M):
            u_ned = C_e_n @ self._channels[ii].unit_vec
            data = [
                self._truth.t(self._tR),
                self._channels[ii].ToW,
                180 + RAD2DEG * np.atan2(u_ned[1], u_ned[0]),  # az should be in ENU not NED
                RAD2DEG * np.asin(u_ned[2]),  # el should be in ENU not NED
                self._channels[ii].phase,
                self._channels[ii].omega,
                self._channels[ii].chip,
                self._channels[ii].chiprate,
                10 * np.log10(self._channels[ii].locks.GetCno()),
                self._channels[ii].E.real,
                self._channels[ii].E.imag,
                self._channels[ii].P.real,
                self._channels[ii].P.imag,
                self._channels[ii].L.real,
                self._channels[ii].L.imag,
                self._channels[ii].P1.real,
                self._channels[ii].P1.imag,
                self._channels[ii].P2.real,
                self._channels[ii].P2.imag,
            ]
            self._chn_files[ii].write(pack("d" * 19, *data))
        return

    def __init_truth_states(self) -> None:
        """
        Calculates true satellite observables based on known user position and receive time
        """
        # parse ephemeris
        eph, atm = ParseEphem(self._conf["ephem_file"])
        self._M = len(eph)
        obs: list[ObservableModel] = []
        for ii in range(self._M):
            obs.append(ObservableModel(eph[ii], atm[ii]))

        # parse truth
        truth = ParseNavSimStates(self._conf["data_file"])
        truth[["lat", "lon", "r", "p", "y"]] *= DEG2RAD
        L = len(truth)

        # initialize output
        tR = np.round(truth["t"].values / 1000.0 + self._conf["init_tow"], 2)
        tT = np.zeros((L, self._M))
        psr = np.zeros((L, self._M))
        psrdot = np.zeros((L, self._M))
        cno = np.zeros((L, self._M))
        self._T_end = tR[-1]

        # open simulation models
        self._cno_model = CnoModel(lamb=LIGHT_SPEED / GPS_L1_FREQUENCY, chip_rate=GPS_CA_CODE_RATE)
        self._clock_model = ClockModel(
            self._conf["clock_model"], self._conf["init_cb"], self._conf["init_cd"]
        )
        self._clock_model._gen = self._gen

        # pregenerate clock states
        cb, cd = self._clock_model.gen(self._conf["sim_dt"], L)
        # cb = np.zeros(L, order="F")
        # cd = np.zeros(L, order="F")

        for kk in range(L):
            # extract known state
            _tR = tR[kk]
            _cb = cb[kk]
            _cd = cd[kk]
            _lla = np.array(
                [truth.loc[kk, "lat"], truth.loc[kk, "lon"], truth.loc[kk, "h"]], order="F"
            )
            _nedv = np.array(
                [truth.loc[kk, "vn"], truth.loc[kk, "ve"], truth.loc[kk, "vd"]], order="F"
            )
            _pos = lla2ecef(_lla)
            _vel = ned2ecefv(_nedv, _lla)

            # iterative Pseudorange/Satellite calculator
            for ii in range(self._M):
                # if kk == 0:
                #     _tT = _tR - 0.07
                # else:
                #     _tT = tT[kk - 1, ii] + self._conf["sim_dt"]

                _d_sv = 100.0
                while _d_sv > 1e-6:
                    _old_sv_pos = obs[ii].SatPos.copy()
                    _psr = obs[ii].Pseudorange
                    _tT = _tR - _psr / LIGHT_SPEED
                    obs[ii].UpdateSatState(_tT)
                    obs[ii].CalcRangeAndRate(_pos, _vel, _cb, _cd, True)
                    _d_sv = np.linalg.norm(_old_sv_pos - obs[ii].SatPos)

                # save satellite state
                psr[kk, ii] = obs[ii].Pseudorange.copy()
                psrdot[kk, ii] = obs[ii].PseudorangeRate.copy()
                tT[kk, ii] = _tR - psr[kk, ii] / LIGHT_SPEED
                # tT[kk, ii] = (
                #     _tR - psr[kk, ii] / LIGHT_SPEED + obs[ii].SatClock[0] - obs[ii].GroupDelay
                # )
                cno[kk, ii] = self._cno_model.sim(obs[ii].Range, self._conf["j2s"])

        # # visualize
        # import matplotlib.pyplot as plt

        # f, ax = plt.subplots()
        # ax.plot(truth["t"], psr[:, 1], label="psr")
        # ax.plot(truth["t"], (tR - tT[:, 1]) * LIGHT_SPEED, label="tR-tT")
        # ax.legend()

        # f2, ax2 = plt.subplots()
        # ax2.plot(truth.loc[1:, "t"], np.diff(psr[:, 2]) / self._conf["sim_dt"], label="diff psr")
        # ax2.plot(truth["t"], psrdot[:, 2], label="psrdot")
        # ax2.legend()

        # plt.show()

        # save truth data to splines (can be sampled at any point)
        self._truth = TruthObservables(
            t=make_splrep(tR, truth["t"].values / 1000.0),
            lat=make_splrep(tR, truth["lat"].values),
            lon=make_splrep(tR, truth["lon"].values),
            h=make_splrep(tR, truth["h"].values),
            vn=make_splrep(tR, truth["vn"].values),
            ve=make_splrep(tR, truth["ve"].values),
            vd=make_splrep(tR, truth["vd"].values),
            r=make_splrep(tR, truth["r"].values),
            p=make_splrep(tR, truth["p"].values),
            y=make_splrep(tR, truth["y"].values),
            cb=make_splrep(tR, cb),
            cd=make_splrep(tR, cd),
            tR=make_splrep(tR, tR),
            tT=[make_splrep(tR, tT[:, ii]) for ii in range(self._M)],
            cno=[make_splrep(tR, cno[:, ii]) for ii in range(self._M)],
            psr=[make_splrep(tR, psr[:, ii]) for ii in range(self._M)],
            psrdot=[make_splrep(tR, psrdot[:, ii]) for ii in range(self._M)],
        )
        return

    def __init_nco(self) -> None:
        """
        Extracts nco states to the nearest sample based on truth initial conditions
        """
        # parse ephemeris
        eph, atm = ParseEphem(self._conf["ephem_file"])

        # initialize each state (must initilize one propagation from beginning)
        tR = self._conf["init_tow"] + 0.02
        tT = np.array([self._truth.tT[ii](tR) for ii in range(self._M)], order="F")
        mod_tT = self.__mod_20_ms(tT) - 0.02
        total_code_phase = self._T_ms * GPS_CA_CODE_LENGTH
        self._channels = []
        delta_samp = []
        _lla = np.array([self._truth.lat(tR), self._truth.lon(tR), self._truth.h(tR)], order="F")
        _pos = lla2ecef(_lla)
        _vel = ned2ecefv(
            np.array([self._truth.vn(tR), self._truth.ve(tR), self._truth.vd(tR)], order="F"), _lla
        )
        _cb = self._truth.cb(tR)
        _cd = self._truth.cd(tR)
        for ii in range(self._M):
            # get perfect nco states
            doppler = -self._truth.psrdot[ii](tR) / self._lambda
            chip_doppler = -(self._truth.psr[ii](tR + 0.005) - self._truth.psr[ii](tR - 0.005)) / (
                self._beta * 0.01
            )
            chip_rate = GPS_CA_CODE_RATE + chip_doppler
            # phase = -self._truth.psr[ii](tR) / self._lambda

            # figure out what sample the channel is at
            code_phase = np.mod(tT[ii], 0.02) * 1000 * GPS_CA_CODE_LENGTH
            code_phase_step = (GPS_CA_CODE_RATE + chip_doppler) / self._conf["samp_freq"]
            current_samp = int(np.ceil(code_phase / code_phase_step))
            total_samp = int(np.ceil(total_code_phase / code_phase_step))
            phase = -self._truth.psr[ii](tR - current_samp / self._conf["samp_freq"]) / self._lambda
            delta_samp.append(total_samp - current_samp)

            # initialize state
            self._channels.append(
                NcoState(
                    ToW=mod_tT[ii],
                    tT=tT[ii],
                    chip=code_phase - current_samp * code_phase_step,
                    chiprate=chip_rate,
                    phase=phase,
                    omega=doppler,
                    current_sample=current_samp,
                    total_sample=total_samp,
                    half_sample=total_samp // 2,
                    model=CorrelatorModel(),
                    obs=ObservableModel(eph[ii], atm[ii]),
                    locks=LockDetectors(0.01),
                    unit_vec=np.zeros(3, order="F"),
                    E=0.0,
                    P=0.0,
                    L=0.0,
                    P1=0.0,
                    P2=0.0,
                )
            )
            _tmp_tT = (
                self._channels[ii].ToW
                + self._channels[ii].chip / self._channels[ii].chiprate
                + self._channels[ii].obs.GroupDelay
            )
            self._channels[ii].obs.UpdateSatState(_tmp_tT)
            self._channels[ii].obs.CalcRangeAndRate(_pos, _vel, _cb, _cd, True)

            # provide correlator model the tap spacing
            self._channels[ii].model._gen = self._gen
            self._channels[ii].model.SetTapSpacing(self._conf["tap_epl"])

        # progress tR forward by largest delta_samps
        self._tR = tR
        self._delta_samp = np.array(delta_samp, order="F")
        self._channel_order = self._delta_samp.argsort()
        return

    @staticmethod
    def __mod_20_ms(x: float | np.ndarray):
        return np.ceil(x / 0.02) * 0.02


if __name__ == "__main__":
    np.set_printoptions(precision=15, linewidth=120)
    sim = VectorTrackingSim("config/vt_correlator_sim.yaml", 2)
    sim.Run()
