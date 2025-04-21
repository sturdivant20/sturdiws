import sys
import numpy as np
from io import BufferedWriter
from pickle import dump, load
from struct import pack
from pathlib import Path
from scipy.interpolate import make_splrep, BSpline
from dataclasses import dataclass
from models import ClockModel, CnoModel, CorrelatorModel, ObservableModel
from navtools import DEG2RAD, RAD2DEG, PI, TWO_PI, LIGHT_SPEED
from navtools._navtools_core.attitude import euler2quat, quat2euler, euler2dcm, dcm2euler
from navtools._navtools_core.frames import lla2ecef, lla2ned, ned2ecefv, ecef2nedDcm
from navtools._navtools_core.math import quatdot, quatinv
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
    tT: list[BSpline] | list[list[BSpline]]
    cno: list[BSpline] | list[list[BSpline]]
    psr: list[BSpline] | list[list[BSpline]]
    psrdot: list[BSpline] | list[list[BSpline]]


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
    unit_vec: np.ndarray[np.double]
    W: np.ndarray[np.double]
    E: np.complex128
    P: np.complex128
    L: np.complex128
    P1: np.complex128
    P2: np.complex128
    P_reg: np.ndarray[np.complex128]

    def __getitem__(self, key):
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            raise KeyError(key)


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
    _ant_body: np.ndarray[np.double]
    _nav_file: BufferedWriter
    _err_file: BufferedWriter
    _chn_files: list[BufferedWriter]

    def __init__(self, file: str | Path, run_index: int = 1, seed: int = None):
        # parse configuration
        self._conf = ParseConfig(file)
        self._T_ms = int(self._conf["meas_dt"] * 1000)
        self._lambda = LIGHT_SPEED / (TWO_PI * GPS_L1_FREQUENCY)
        self._beta = LIGHT_SPEED / GPS_CA_CODE_RATE
        self._kappa = GPS_CA_CODE_RATE / (TWO_PI * GPS_L1_FREQUENCY)
        self._intmd_freq = TWO_PI * self._conf["intmd_freq"]

        # random number generator
        self._gen = np.random.default_rng(seed)

        # file path
        mypath = (
            Path(self._conf["out_folder"])
            / self._conf["scenario"]
            / f"CNo_{self._conf["cno"]}_dB"
            / f"Run{run_index}"
        )
        mypath.mkdir(parents=True, exist_ok=True)

        # apply correct functionality based on antenna configuration
        if self._conf["is_multi_antenna"]:
            filename = mypath / "truth_splines_array.bin"
            ant_body = []
            for jj in range(self._conf["n_ant"]):
                ant_body.append(self._conf[f"ant_xyz_{jj}"])
            self._ant_body = np.array(ant_body, order="F").T
            self._correlation_scheme = self.__vt_array_correlate
            self._vector_update = self.__vt_array_update
        else:
            filename = mypath / "truth_splines.bin"
            self._correlation_scheme = self.__vt_correlate
            self._vector_update = self.__vt_update

        # generate truth observables
        self.__init_truth_states()
        # if filename.exists():
        #     with open(filename, "rb") as file:
        #         self._truth = load(file)
        #     self._clock_model = ClockModel(
        #         self._conf["clock_model"], self._conf["init_cb"], self._conf["init_cd"]
        #     )
        #     self._T_end = self._truth.t.t[-1]
        # else:
        #     self.__init_truth_states()
        #     with open(filename, "wb") as file:
        #         dump(self._truth, file)

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
            self._truth.r(self._tR),  # - 0.4,
            self._truth.p(self._tR),  # - 0.26,
            self._truth.y(self._tR),  # + 1,
            self._truth.cb(self._tR) * LIGHT_SPEED,
            self._truth.cd(self._tR) * LIGHT_SPEED,
        )
        self._kf.SetClockSpec(
            self._clock_model._model.h0, self._clock_model._model.h1, self._clock_model._model.h2
        )
        self._kf.SetProcessNoise(self._conf["vel_process_psd"], self._conf["att_process_psd"])

        # open files for binary output
        self._nav_file = open(mypath / "Nav_Results_Log.bin", "wb")
        self._err_file = open(mypath / "Err_Results_Log.bin", "wb")
        self._chn_files = [
            open(mypath / f"Channel_{i}_Results_Log.bin", "wb") for i in range(self._M)
        ]
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
        next_tR = self._tR + self._conf["meas_dt"]
        while next_tR < self._T_end:

            # run an update from each channel
            for ii in self._channel_order:
                # correlate first and second halves
                self._correlation_scheme(ii)

                # run lock detectors
                self._channels[ii].locks.Update(self._channels[ii].P, self._conf["meas_dt"])

                # update channel states
                self._channels[ii].ToW += self._conf["meas_dt"]
                self._channels[ii].ToW = np.round(self._channels[ii].ToW, 2)
                self._channels[ii].chip -= self._T_ms * GPS_CA_CODE_LENGTH

                # vector update
                d_samp = self._channels[ii].total_sample - self._channels[ii].current_sample
                self._vector_update(self._channels[ii], d_samp)

                # move all other channels "current_sample" forward
                for jj in range(self._M):
                    self._channels[jj].current_sample += d_samp
                self._channels[ii].current_sample = 0

                # use vector nco frequencies to predict next update
                self._channels[ii].total_sample, self._channels[ii].half_sample = (
                    self.__new_code_period(self._channels[ii].chiprate, self._channels[ii].chip)
                )

                # increment
                next_tR = self._tR + self._conf["meas_dt"]

            # log results to binary file
            # print("---------------------------------------------------------------------------")
            self.__update_processing_order()
            self.__log_to_file()
        return

    def __new_code_period(self, chiprate: np.double, rem_code_phase: np.double) -> tuple[int, int]:
        """
        Predicts the number of samples required to complete the next code period
        """
        code_phase_step = chiprate / self._conf["samp_freq"]
        total_samp = int((self._T_ms * GPS_CA_CODE_LENGTH - rem_code_phase) / code_phase_step)
        half_samp = total_samp // 2
        return total_samp, half_samp

    def __vt_update(self, channel: NcoState, delta_samp: int) -> None:
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
        psr_err, _psr_var, psrdot_err, _psrdot_var = self.__vt_discriminators(channel)

        # 4. Estimate tracking psr and psrdot (combine vector and scalar)
        _psr = -psr_err + LIGHT_SPEED * (_tR - _tT)
        _psrdot = -psrdot_err - self._lambda * channel.omega + LIGHT_SPEED * channel.obs.SatClock[1]

        # 5. Call vector processing update
        self._kf.GnssUpdate(
            channel.obs.SatPos, channel.obs.SatVel, _psr, _psrdot, _psr_var, _psrdot_var
        )

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

        # 7. Vector NCO update
        channel.chiprate = (GPS_CA_CODE_RATE * self._conf["meas_dt"] - channel.chip) / (
            _tR_pred - _tR
        )
        channel.omega = -(channel.obs.RangeRate + _cd_pred) / self._lambda
        channel.unit_vec = channel.obs.EcefUnitVec.copy()
        self._tR = _tR
        return

    def __vt_array_update(self, channel: NcoState, delta_samp: int):
        """
        Runs an antenna array Vector Delay-Frequency Lock Loop measurement update
        """
        # 1. Update satellite pos, vel, and clock terms from transmit time
        _tT = (
            channel.ToW
            + channel.chip / channel.chiprate
            + channel.obs.GroupDelay
            - channel.obs.SatClock[0]
        )
        channel.obs.UpdateSatState(_tT)
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
        psr_err, _psr_var, psrdot_err, _psrdot_var, _delta_phase, _phase_var = (
            self.__vt_array_discriminators(channel)
        )

        # 4. Estimate tracking psr and psrdot (combine vector and scalar)
        _psr = -psr_err + LIGHT_SPEED * (_tR - _tT)
        _psrdot = -psrdot_err - self._lambda * channel.omega + LIGHT_SPEED * channel.obs.SatClock[1]

        # 5. Call vector processing update
        # self._kf.GnssUpdate(
        #     channel.obs.SatPos, channel.obs.SatVel, _psr, _psrdot, _psr_var, _psrdot_var
        # )
        self._kf.PhasedArrayUpdate(
            channel.obs.SatPos,
            channel.obs.SatVel,
            _psr,
            _psrdot,
            _delta_phase,
            _psr_var,
            _psrdot_var,
            _phase_var,
            self._ant_body,
            self._conf["n_ant"],
            self._lambda,
        )

        # 6. Predict ECEF state at end of next code period
        _lla = np.array([self._kf.phi_, self._kf.lam_, self._kf.h_], order="F")
        _nedv = np.array([self._kf.vn_, self._kf.ve_, self._kf.vd_], order="F")
        _vel_pred = ned2ecefv(_nedv, _lla)
        _pos_pred = lla2ecef(_lla) + _vel_pred * self._conf["meas_dt"]
        _cd_pred = self._kf.cd_
        _cb_pred = self._kf.cb_ + _cd_pred * self._conf["meas_dt"]
        _tT_pred = (
            channel.ToW + self._conf["meas_dt"] + channel.obs.GroupDelay - channel.obs.SatClock[0]
        )
        channel.obs.UpdateSatState(_tT_pred)
        channel.obs.CalcRangeAndRate(
            _pos_pred, _vel_pred, _cb_pred / LIGHT_SPEED, _cd_pred / LIGHT_SPEED, True
        )
        _tR_pred = channel.ToW + self._conf["meas_dt"] + channel.obs.Pseudorange / LIGHT_SPEED

        # 7. Vector NCO update
        channel.chiprate = (GPS_CA_CODE_RATE * self._conf["meas_dt"] - channel.chip) / (
            _tR_pred - _tR
        )
        channel.omega = -(channel.obs.RangeRate + _cd_pred) / self._lambda
        channel.unit_vec = channel.obs.EcefUnitVec.copy()
        self._tR = _tR
        return

    def __vt_discriminators(
        self, channel: NcoState
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """
        Calculates the code and frequency tracking errors
        """
        cno = channel.locks.GetCno()
        psr_err = np.array([self._beta * DllNneml2(channel.E, channel.L)], order="F")
        psr_var = np.array([self._beta**2 * DllVariance(cno, self._conf["meas_dt"])], order="F")
        psrdot_err = np.array(
            [self._lambda * FllAtan2(channel.P1, channel.P2, self._conf["meas_dt"])], order="F"
        )
        psrdot_var = np.array(
            [self._lambda**2 * FllVariance(cno, self._conf["meas_dt"])], order="F"
        )
        # print(f"psr_err = {psr_err} | psrdot_err = {psrdot_err}")
        return psr_err, psr_var, psrdot_err, psrdot_var

    def __vt_array_discriminators(self, channel: NcoState) -> tuple[
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
        np.ndarray[float],
    ]:
        """
        Calculates the code and frequency tracking errors, along with phased array discriminators
        """
        cno = channel.locks.GetCno()
        psr_err = np.array([self._beta * DllNneml2(channel.E, channel.L)], order="F")
        psr_var = np.array([self._beta**2 * DllVariance(cno, self._conf["meas_dt"])], order="F")
        psrdot_err = np.array(
            [self._lambda * FllAtan2(channel.P1, channel.P2, self._conf["meas_dt"])], order="F"
        )
        psrdot_var = np.array(
            [self._lambda**2 * FllVariance(cno, self._conf["meas_dt"])], order="F"
        )

        phase_err = np.zeros(self._conf["n_ant"], order="F")
        for jj in range(self._conf["n_ant"]):
            phase_err[jj] = PllAtan2(channel.P_reg[jj])
        # print(f"phase_err = {phase_err}")
        phase_err = np.fmod(phase_err - phase_err[0] + PI, TWO_PI) - PI
        phase_err[phase_err > PI] -= TWO_PI
        phase_err[phase_err < -PI] += TWO_PI
        phase_var = (
            2.0
            * PllVariance(cno / self._conf["n_ant"], self._conf["meas_dt"])
            * np.ones(self._conf["n_ant"], order="F")
        )
        # print(f"phase_err = {phase_err}")
        return psr_err, psr_var, psrdot_err, psrdot_var, phase_err, phase_var

    def __vt_correlate(self, ii: int) -> None:
        """
        Correlates channel 'ii' to the true signal
        """
        self._channels[ii].E = 0.0
        self._channels[ii].P = 0.0
        self._channels[ii].L = 0.0
        self._channels[ii].P1 = 0.0
        self._channels[ii].P2 = 0.0

        T = self._tR - self._channels[ii].current_sample / self._conf["samp_freq"]
        for kk in range(2):
            if kk:
                dt = (
                    self._channels[ii].total_sample - self._channels[ii].half_sample
                ) / self._conf["samp_freq"]
            else:
                dt = self._channels[ii].half_sample / self._conf["samp_freq"]

            # integrate
            T += dt
            self._channels[ii].phase += self._channels[ii].omega * dt
            self._channels[ii].chip += self._channels[ii].chiprate * dt

            # truth
            omega = -self._truth.psrdot[ii](T) / self._lambda
            phase = -self._truth.psr[ii](T) / self._lambda
            chiprate = GPS_CA_CODE_RATE - (
                self._truth.psr[ii](T + 0.005) - self._truth.psr[ii](T - 0.005)
            ) / (self._beta * 0.01)
            chip = np.mod(self._truth.tT[ii](T), 0.02) * 1000 * GPS_CA_CODE_LENGTH

            # correlate
            R = self._channels[ii].model.Correlate(
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
            self._channels[ii].E += R[0]
            self._channels[ii].P += R[1]
            self._channels[ii].L += R[2]
            self._channels[ii][f"P{kk+1}"] += R[1]
        return

    def __vt_array_correlate(self, ii: int) -> None:
        """
        Correlates channel 'ii' to 'n_ant' antenna array feeds
        """
        self._channels[ii].E = 0.0
        self._channels[ii].P = 0.0
        self._channels[ii].L = 0.0
        self._channels[ii].P1 = 0.0
        self._channels[ii].P2 = 0.0
        self._channels[ii].P_reg[:] = 0.0

        # beamsteer
        self.__beamsteer(self._channels[ii])
        # print(self._channels[ii].W)

        T = self._tR - self._channels[ii].current_sample / self._conf["samp_freq"]
        for kk in range(2):
            if kk:
                dt = (
                    self._channels[ii].total_sample - self._channels[ii].half_sample
                ) / self._conf["samp_freq"]
            else:
                dt = self._channels[ii].half_sample / self._conf["samp_freq"]

            # integrate
            T += dt
            self._channels[ii].phase += self._channels[ii].omega * dt
            self._channels[ii].chip += self._channels[ii].chiprate * dt

            # truth
            omega = np.zeros(self._conf["n_ant"], order="F")
            phase = np.zeros(self._conf["n_ant"], order="F")
            chiprate = np.zeros(self._conf["n_ant"], order="F")
            chip = np.zeros(self._conf["n_ant"], order="F")
            for jj in range(self._conf["n_ant"]):
                omega[jj] = -self._truth.psrdot[ii][jj](T) / self._lambda
                phase[jj] = -self._truth.psr[ii][jj](T) / self._lambda
                chiprate[jj] = GPS_CA_CODE_RATE - (
                    self._truth.psr[ii][jj](T + 0.005) - self._truth.psr[ii][jj](T - 0.005)
                ) / (self._beta * 0.01)
                chip[jj] = np.mod(self._truth.tT[ii][jj](T), 0.02) * 1000 * GPS_CA_CODE_LENGTH

            # correlate
            R = self._channels[ii].model.CorrelateArray(
                dt,
                self._truth.cno[ii][0](T),
                chip,
                chiprate,
                phase,
                omega,
                self._channels[ii].chip,
                self._channels[ii].chiprate,
                self._channels[ii].phase,
                self._channels[ii].omega,
            )
            self._channels[ii].E += self._channels[ii].W @ R[:, 0]
            self._channels[ii].L += self._channels[ii].W @ R[:, 2]
            Prompt = self._channels[ii].W @ R[:, 1]
            self._channels[ii].P += Prompt
            self._channels[ii][f"P{kk+1}"] = Prompt
            self._channels[ii].P_reg += R[:, 1]
            # self._channels[ii].E += R[0, 0]
            # self._channels[ii].P += R[0, 1]
            # self._channels[ii].L += R[0, 2]
            # self._channels[ii][f"P{kk+1}"] += R[0, 1]
        return

    def __beamsteer(self, channel: NcoState):
        """
        Deterministic beamsteering equation
        """
        # _lla = np.array(
        #     [self._truth.lat(self._tR), self._truth.lon(self._tR), self._truth.h(self._tR)],
        #     order="F",
        # )
        # _C_e_n = ecef2nedDcm(_lla)
        # _rpy = np.array(
        #     [self._truth.r(self._tR), self._truth.p(self._tR), self._truth.y(self._tR)], order="F"
        # )
        # _C_l_b = euler2dcm(_rpy).T
        # u_body = _C_l_b @ (_C_e_n @ channel.unit_vec)
        _lla = np.array([self._kf.phi_, self._kf.lam_, self._kf.h_], order="F")
        _C_e_n = ecef2nedDcm(_lla)
        u_body = self._kf.C_b_l_.T @ (_C_e_n @ channel.unit_vec)
        channel.W = np.exp(1j / self._lambda * (self._ant_body.T @ u_body))
        return

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
        rpy_err = quat2euler(quatdot(q_true, quatinv(self._kf.q_b_l_)), True) * RAD2DEG
        # C_true = euler2dcm(rpy_true)
        # rpy_err = dcm2euler(C_true @ self._kf.C_b_l_.T, True) * RAD2DEG

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
        self._nav_file.flush()
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
        self._err_file.flush()

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
            self._chn_files[ii].flush()
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
        tR = np.round(truth["t"].values / 1000.0 + self._conf["init_tow"], 2)
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

        if self._conf["is_multi_antenna"]:
            # initialize output
            tT = np.zeros((L, self._M, self._conf["n_ant"]), order="F")
            psr = np.zeros((L, self._M, self._conf["n_ant"]), order="F")
            psrdot = np.zeros((L, self._M, self._conf["n_ant"]), order="F")
            cno = np.zeros((L, self._M, self._conf["n_ant"]), order="F")

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
                _rpy = np.array(
                    [truth.loc[kk, "r"], truth.loc[kk, "p"], truth.loc[kk, "y"]], order="F"
                )
                _C_n_e = ecef2nedDcm(_lla).T
                _C_b_n = euler2dcm(_rpy, True)
                _xyz = lla2ecef(_lla)
                _xyzv = _C_n_e @ _nedv
                _ant_xyz = _xyz[:, None] + _C_n_e @ (_C_b_n @ self._ant_body)

                # iterative Pseudorange/Satellite calculator
                for ii in range(self._M):

                    # loop through antennas
                    for jj in range(self._conf["n_ant"]):
                        _d_sv = 100.0
                        while _d_sv > 1e-6:
                            _old_sv_pos = obs[ii].SatPos.copy()
                            _psr = obs[ii].Pseudorange
                            _tT = _tR - _psr / LIGHT_SPEED
                            obs[ii].UpdateSatState(_tT)
                            obs[ii].CalcRangeAndRate(_ant_xyz[:, jj], _xyzv, _cb, _cd, True)
                            _d_sv = np.linalg.norm(_old_sv_pos - obs[ii].SatPos)

                        # save satellite state
                        psr[kk, ii, jj] = obs[ii].Pseudorange.copy()
                        psrdot[kk, ii, jj] = obs[ii].PseudorangeRate.copy()
                        tT[kk, ii, jj] = _tR - psr[kk, ii, jj] / LIGHT_SPEED
                        cno[kk, ii, jj] = self._cno_model.sim(obs[ii].Range, self._conf["j2s"])

            # save truth data to splines (can be sampled at any point)
            self._truth = TruthObservables(
                t=make_splrep(tR, truth["t"].values / 1000.0),
                lat=make_splrep(tR, truth["lat"].values),
                lon=make_splrep(tR, truth["lon"].values),
                h=make_splrep(tR, truth["h"].values),
                vn=make_splrep(tR, truth["vn"].values, k=5),
                ve=make_splrep(tR, truth["ve"].values, k=5),
                vd=make_splrep(tR, truth["vd"].values, k=5),
                r=make_splrep(tR, truth["r"].values, k=5),
                p=make_splrep(tR, truth["p"].values, k=5),
                y=make_splrep(tR, np.unwrap(truth["y"].values), k=5),
                cb=make_splrep(tR, cb, k=5),
                cd=make_splrep(tR, cd, k=5),
                tR=make_splrep(tR, tR),
                tT=[
                    [make_splrep(tR, tT[:, ii, jj]) for jj in range(self._conf["n_ant"])]
                    for ii in range(self._M)
                ],
                cno=[
                    [make_splrep(tR, cno[:, ii, jj]) for jj in range(self._conf["n_ant"])]
                    for ii in range(self._M)
                ],
                psr=[
                    [make_splrep(tR, psr[:, ii, jj]) for jj in range(self._conf["n_ant"])]
                    for ii in range(self._M)
                ],
                psrdot=[
                    [make_splrep(tR, psrdot[:, ii, jj]) for jj in range(self._conf["n_ant"])]
                    for ii in range(self._M)
                ],
            )
        else:
            # initialize output
            tT = np.zeros((L, self._M), order="F")
            psr = np.zeros((L, self._M), order="F")
            psrdot = np.zeros((L, self._M), order="F")
            cno = np.zeros((L, self._M), order="F")

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
                _xyz = lla2ecef(_lla)
                _xyzv = ned2ecefv(_nedv, _lla)

                # iterative Pseudorange/Satellite calculator
                for ii in range(self._M):
                    _d_sv = 100.0
                    while _d_sv > 1e-6:
                        _old_sv_pos = obs[ii].SatPos.copy()
                        _psr = obs[ii].Pseudorange
                        _tT = _tR - _psr / LIGHT_SPEED
                        obs[ii].UpdateSatState(_tT)
                        obs[ii].CalcRangeAndRate(_xyz, _xyzv, _cb, _cd, True)
                        _d_sv = np.linalg.norm(_old_sv_pos - obs[ii].SatPos)

                    # save satellite state
                    psr[kk, ii] = obs[ii].Pseudorange.copy()
                    psrdot[kk, ii] = obs[ii].PseudorangeRate.copy()
                    tT[kk, ii] = _tR - psr[kk, ii] / LIGHT_SPEED
                    cno[kk, ii] = self._cno_model.sim(obs[ii].Range, self._conf["j2s"])

            # save truth data to splines (can be sampled at any point)
            self._truth = TruthObservables(
                t=make_splrep(tR, truth["t"].values / 1000.0),
                lat=make_splrep(tR, truth["lat"].values),
                lon=make_splrep(tR, truth["lon"].values),
                h=make_splrep(tR, truth["h"].values),
                vn=make_splrep(tR, truth["vn"].values, k=5),
                ve=make_splrep(tR, truth["ve"].values, k=5),
                vd=make_splrep(tR, truth["vd"].values, k=5),
                r=make_splrep(tR, truth["r"].values, k=5),
                p=make_splrep(tR, truth["p"].values, k=5),
                y=make_splrep(tR, np.unwrap(truth["y"].values), k=5),
                cb=make_splrep(tR, cb, k=5),
                cd=make_splrep(tR, cd, k=5),
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
        self._M = len(eph)

        # initialize each state (must initilize one propagation from beginning)
        tR = self._conf["init_tow"] + 0.02
        if self._conf["is_multi_antenna"]:
            tT = np.array([self._truth.tT[ii][0](tR) for ii in range(self._M)], order="F")
        else:
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
            if self._conf["is_multi_antenna"]:
                doppler = -self._truth.psrdot[ii][0](tR) / self._lambda
                chip_doppler = -(
                    self._truth.psr[ii][0](tR + 0.005) - self._truth.psr[ii][0](tR - 0.005)
                ) / (self._beta * 0.01)
            else:
                doppler = -self._truth.psrdot[ii](tR) / self._lambda
                chip_doppler = -(
                    self._truth.psr[ii](tR + 0.005) - self._truth.psr[ii](tR - 0.005)
                ) / (self._beta * 0.01)
            chip_rate = GPS_CA_CODE_RATE + chip_doppler

            # figure out what sample the channel is at
            code_phase = np.mod(tT[ii], 0.02) * 1000 * GPS_CA_CODE_LENGTH
            code_phase_step = chip_rate / self._conf["samp_freq"]
            current_samp = int(np.ceil(code_phase / code_phase_step))
            total_samp = int(np.ceil(total_code_phase / code_phase_step))
            if self._conf["is_multi_antenna"]:
                phase = (
                    -self._truth.psr[ii][0](tR - current_samp / self._conf["samp_freq"])
                    / self._lambda
                )
            else:
                phase = (
                    -self._truth.psr[ii](tR - current_samp / self._conf["samp_freq"]) / self._lambda
                )
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
                    locks=LockDetectors(0.005),
                    unit_vec=np.zeros(3, order="F"),
                    W=np.zeros(self._conf["n_ant"], order="F"),
                    E=0.0,
                    P=0.0,
                    L=0.0,
                    P1=0.0,
                    P2=0.0,
                    P_reg=np.zeros(self._conf["n_ant"], order="F", dtype=np.complex128),
                )
            )
            _tmp_tT = (
                self._channels[ii].ToW
                + self._channels[ii].chip / self._channels[ii].chiprate
                + self._channels[ii].obs.GroupDelay
            )
            self._channels[ii].obs.UpdateSatState(_tmp_tT)
            self._channels[ii].obs.CalcRangeAndRate(_pos, _vel, _cb, _cd, True)
            self._channels[ii].unit_vec = self._channels[ii].obs.EcefUnitVec.copy()

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
    from time import time

    t0 = time()
    np.set_printoptions(precision=6, linewidth=120)
    sim = VectorTrackingSim(
        "config/vt_correlator_sim.yaml", 1, 226407869803896429276743162746548480267
    )
    sim.Run()
    print(f"Total time: {(time() - t0):.3f} s")
