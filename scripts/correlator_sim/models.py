import numpy as np
from datetime import datetime, timedelta
from navtools import LIGHT_SPEED, PI, PI_SQU, BOLTZMANN, WGS84_OMEGA, WGS84_OMEGA_SKEW
from navtools._navtools_core.frames import lla2ecef, ned2ecefv
from sturdins._sturdins_core.navsense import GetNavClock, NavigationClock
from satutils._satutils_core.atmosphere import TropoModel, IonoModel, KlobucharElements
from satutils._satutils_core.ephemeris import KeplerEphem, KeplerElements
from satutils import GPS_CA_CODE_LENGTH


def gps_to_utc_time(week, tow):
    # GPS Epoch: January 6, 1980
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)

    # Calculate the total seconds since the GPS epoch
    total_seconds = week * 604800 + tow

    # convert to utc time (-18 leap seconds)
    return gps_epoch + timedelta(seconds=total_seconds - 18)


class ClockModel:
    """
    Atomic clock simulation tool
    """

    _x: np.ndarray[np.double]
    _Phi: np.ndarray[np.double]
    _Q: np.ndarray[np.double]
    _Sigma: np.ndarray[np.double]
    _model: NavigationClock
    _method: callable
    _method_str: str
    _gen: np.random.Generator
    _Sb: np.double
    _Sf: np.double
    _Sd: np.double
    _init_x: np.ndarray[np.double]

    def __init__(
        self,
        model: str = "high_quality_tcxo",
        init_cb: float = 0.0,
        init_cd: float = 0.0,
        method: str = "svd",
        seed: int = None,
    ):
        self._model = GetNavClock(model)
        match method.casefold():
            case "svd":
                self._method = self.__gen_svd
                self._method_str = "svd"
            case "eigh":
                self._method = self.__gen_eigh
                self._method_str = "eigh"
            case "cholesky":
                self._method = self.__gen_cholesky
                self._method_str = "cholesky"
        self._x = np.array([init_cb, init_cd], order="F")
        self._init_x = np.array([init_cb, init_cd], order="F")
        self._Sb = self._model.h0 / 2.0
        self._Sf = self._model.h1 * 2.0
        self._Sd = self._model.h2 * 2.0 * PI_SQU
        self._gen = np.random.default_rng(seed=seed)
        pass

    def sim(self, T: float):
        self.__gen_matrices(T)
        self._x = self._Phi @ self._x + self._Sigma @ self._gen.standard_normal(2)
        return self._x

    def gen(self, T: float, n_samp: int):
        self.__gen_matrices(T)
        mu = np.zeros(2, order="F")
        x, y = self._gen.multivariate_normal(
            mean=mu, cov=self._Q, size=n_samp, method=self._method_str
        ).T
        drift = np.cumsum(y) + self._init_x[1]
        bias = np.cumsum(x) + np.cumsum(drift * T) + self._init_x[0]
        return bias, drift

    def __gen_matrices(self, T):
        self._Phi = np.array([[1, T], [0, 1]], order="F")
        # q_bb = self._Sb * T + self._Sd * T**3 / 3
        # q_bd = self._Sd * T**2 / 2
        # q_dd = self._Sd * T
        q_bb = self._Sb * T + self._Sf * T**2 + self._Sd * T**3 / 3
        q_bd = self._Sf * T + self._Sd * T**2 / 2
        q_dd = self._Sb / T + self._Sf + self._Sd * T
        self._Q = np.array([[q_bb, q_bd], [q_bd, q_dd]], order="F")
        self._Sigma = self._method(self._Q)

    @staticmethod
    def __gen_svd(Q: np.ndarray[np.double]):
        U, S, _ = np.linalg.svd(Q)
        return U @ np.diag(np.sqrt(S))

    @staticmethod
    def __gen_eigh(Q: np.ndarray[np.double]):
        d, W = np.linalg.eigh(Q)
        return W @ np.diag(np.sqrt(d))

    @staticmethod
    def __gen_cholesky(Q: np.ndarray[np.double]):
        return np.linalg.cholesky(Q)


#! ---------------------------------------------------------------------------------------------- !#


class CnoModel:
    """
    Simple free-space path loss model under jamming
    """

    _temp: np.double
    _lamb: np.double
    _chip_rate: np.double
    _scale_factor: np.double
    _C: np.double
    _N0: np.double

    def __init__(
        self, temp: float = 300.0, lamb: float = LIGHT_SPEED / 1575.42e6, chip_rate: float = 1.023e6
    ):
        self._temp = temp
        self._lamb = lamb
        self._chip_rate = chip_rate
        self._scale_factor = 2.22  # assumes BPSK-BLWN

        self._N0 = BOLTZMANN * temp
        self._C = 10 ** (14.25 / 10) * 10 ** (13.5 / 10) / 10 ** (1.25 / 10)  # GPS L1 C/A power

    def sim(self, ranges: float | np.ndarray[float], J2S: float = None):
        C = self._C * (self._lamb / (2 * PI * ranges)) ** 2
        cno = C / self._N0
        if J2S is not None:
            j2s = 10 ** (J2S / 10)
            cno = 1 / ((1 / cno) + (j2s / (self._scale_factor * self._chip_rate)))
        return cno


#! ---------------------------------------------------------------------------------------------- !#


class ObservableModel:
    """
    Simple model to produce GPS observables
    """

    _sv_eph: KeplerEphem
    _sv_elem: KeplerElements
    _sv_iono: IonoModel
    _sv_tropo: TropoModel
    _sv_clk: np.ndarray[np.double]
    _sv_pos: np.ndarray[np.double]
    _sv_vel: np.ndarray[np.double]
    _sv_acc: np.ndarray[np.double]
    _cb: np.double
    _cd: np.double
    _I: np.double
    _T: np.double
    _r: np.double
    _rr: np.double
    _u: np.ndarray[np.double]

    def __init__(self, elem: KeplerElements, atm: KlobucharElements):
        self._sv_elem = elem
        self._sv_eph = KeplerEphem(elem)
        self._sv_iono = IonoModel(atm)
        self._sv_tropo = TropoModel()
        self._sv_clk = np.zeros(3, order="F")
        self._sv_pos = np.zeros(3, order="F")
        self._sv_vel = np.zeros(3, order="F")
        self._sv_acc = np.zeros(3, order="F")
        self._cb = 0.0
        self._cd = 0.0
        self._I = 0.0
        self._T = 0.0
        self._r = 0.0
        self._rr = 0.0
        self._u = np.zeros(3, order="F")

    @property
    def Range(self):
        return self._r

    @property
    def Pseudorange(self):
        return self._r + LIGHT_SPEED * (
            self._cb + self._sv_elem.tgd - self._sv_clk[0] + self._I + self._T
        )

    @property
    def PhasePseudorange(self):
        return self._r + LIGHT_SPEED * (
            self._cb + self._sv_elem.tgd - self._sv_clk[0] - self._I + self._T
        )

    @property
    def RangeRate(self):
        return self._rr

    @property
    def PseudorangeRate(self):
        return self._rr + LIGHT_SPEED * (self._cd - self._sv_clk[1])

    @property
    def SatPos(self):
        return self._sv_pos

    @property
    def SatVel(self):
        return self._sv_vel

    @property
    def SatClock(self):
        return self._sv_clk

    @property
    def GroupDelay(self):
        return self._sv_elem.tgd

    @property
    def EcefUnitVec(self):
        return self._u

    def UpdateSatState(self, tT: float):
        self._sv_eph.CalcNavStates(self._sv_clk, self._sv_pos, self._sv_vel, self._sv_acc, tT)

    def UpdateAtmState(self, lla: np.ndarray[float], az: float, el: float, week: int, tow: float):
        self._I = self._sv_iono.CalcIonoDelay(tow, lla[0], lla[1], az, el)
        doy = gps_to_utc_time(week, tow).timetuple().tm_yday
        self._T = self._sv_tropo.CalcTropoDelay(doy, lla[0], lla[2], el)

    def CalcRangeAndRate(
        self,
        pos: np.ndarray[float],
        vel: np.ndarray[float],
        cb: float,
        cd: float,
        is_ecef: bool = True,
    ):
        if not is_ecef:
            vel = ned2ecefv(vel, pos)
            pos = lla2ecef(pos)

        self._cb = cb
        self._cd = cd

        # account for earth's rotation
        dr = pos - self._sv_pos
        r = np.linalg.norm(dr)
        wt = WGS84_OMEGA * r / LIGHT_SPEED
        swt = np.sin(wt)
        cwt = np.cos(wt)
        C_I_e = np.array([[cwt, swt, 0], [-swt, cwt, 0], [0, 0, 1]], order="F")

        # predict range
        dr = pos - C_I_e @ self._sv_pos
        self._r = np.linalg.norm(dr)
        self._u = dr / self._r

        # predict range-rate
        dv = (vel + WGS84_OMEGA_SKEW @ pos) - C_I_e @ (
            self._sv_vel + WGS84_OMEGA_SKEW @ self._sv_pos
        )
        self._rr = self._u.dot(dv)


#! ---------------------------------------------------------------------------------------------- !#


class CorrelatorModel:
    """
    Simple model to produce GPS tracking loop correlation results
    """

    _gen: np.random.Generator
    _sqrtN: np.double
    _I: np.ndarray[np.complex128]
    _Sigma: np.ndarray[np.complex128]
    _SigmaSq: np.ndarray[np.complex128]
    _method: callable
    _method_str: str
    _tap_space: np.ndarray[np.double]

    def __init__(self, method: str = "svd", seed: int = None):
        self._gen = np.random.default_rng(seed)
        self._sqrtN = np.sqrt(2)
        self._avg_cno = 0.0
        match method.casefold():
            case "svd":
                self._method = self.__gen_svd
                self._method_str = "svd"
            case "eigh":
                self._method = self.__gen_eigh
                self._method_str = "eigh"
            case "cholesky":
                self._method = self.__gen_cholesky
                self._method_str = "cholesky"

    def SetTapSpacing(self, tap_space: float) -> None:
        self._tap_space = np.array([-tap_space, 0.0, tap_space], order="F")
        Iep = self.CalculateI(1.0, -tap_space, 0.0, 0.0, 0.0)
        Ipp = self.CalculateI(1.0, 0.0, 0.0, 0.0, 0.0)
        Ipl = self.CalculateI(1.0, tap_space, 0.0, 0.0, 0.0)
        Iel = self.CalculateI(1.0, 2 * tap_space, 0.0, 0.0, 0.0)
        self._SigmaSq = np.array(
            [[Ipp, np.conj(Iep), Iel], [Iep, Ipp, np.conj(Ipl)], [np.conj(Iel), Ipl, Ipp]],
            order="F",
        )
        self._Sigma = self._method(self._SigmaSq)
        self._I = np.zeros(3, dtype=np.complex128, order="F")

    def Correlate(
        self,
        T: float,
        cno: float,
        chip: float,
        chiprate: float,
        phase: float,
        omega: float,
        est_chip: float,
        est_chiprate: float,
        est_phase: float,
        est_omega: float,
    ) -> None:
        chip_err = chip - est_chip
        chip_rate_err = chiprate - est_chiprate
        phase_err = phase - est_phase
        omega_err = omega - est_omega

        thresh = T * 1000 * GPS_CA_CODE_LENGTH
        if chip_err < -thresh:
            chip_err += 2 * thresh
        elif chip_err > thresh:
            chip_err -= 2 * thresh
        # print(f"chip|rate err = {chip_err} | {chip_rate_err}")
        # print(f"chip_err = {chip_err}")
        # print(f"chip_rate_err = {chip_rate_err}")
        # print(f"phase_err = {phase_err}")
        # print(f"omega_err = {omega_err}")

        I = np.zeros(3, order="F", dtype=np.complex128)
        for ii in range(3):
            I[ii] += self.CalculateI(
                T,
                chip_err + self._tap_space[ii],
                chip_rate_err,
                phase_err,
                omega_err,
            )

        n = self._gen.standard_normal(3) + 1j * self._gen.standard_normal(3)
        A = 2.0 / self._sqrtN * np.sqrt(cno / T)
        return (A * I) + (self._Sigma @ n)

    def CorrelateArray(
        self,
        T: float,
        cno: float,
        chip: np.ndarray[float],
        chiprate: np.ndarray[float],
        phase: np.ndarray[float],
        omega: float,
        est_chip: float,
        est_chiprate: float,
        est_phase: float,
        est_omega: float,
    ):
        # return self.Correlate(
        #     T,
        #     cno,
        #     chip[0],
        #     chiprate[0],
        #     phase[0],
        #     omega[0],
        #     est_chip,
        #     est_chiprate,
        #     est_phase,
        #     est_omega,
        # )
        chip_err = chip - est_chip
        chip_rate_err = chiprate - est_chiprate
        phase_err = phase - est_phase
        omega_err = omega - est_omega

        thresh = 10 * GPS_CA_CODE_LENGTH
        chip_err[chip_err < -thresh] += 2 * thresh
        chip_err[chip_err > thresh] -= 2 * thresh
        # print(f"chip|rate err = {chip_err} | {chip_rate_err}")

        # L = 3 * phase.size
        # SigmaSq = np.zeros((L, L), order="F", dtype=np.complex128)
        # for ii, _ii in zip(range(phase.size), range(0, L, 3)):
        #     for jj, _jj in zip(range(phase.size), range(0, L, 3)):
        #         SigmaPhaseSq = self.CalculateI(1.0, 0.0, 0.0, phase[ii] - phase[jj], 0.0)
        #         SigmaSq[_ii : _ii + 3, _jj : _jj + 3] = SigmaPhaseSq * self._SigmaSq
        # Sigma = self._method(SigmaSq)

        # I = np.zeros(L, order="F", dtype=np.complex128)
        # kk = 0
        # for jj in range(chip.size):
        #     for ii in range(3):
        #         I[kk] = self.CalculateI(
        #             T,
        #             chip_err[jj] + self._tap_space[ii],
        #             chip_rate_err[jj],
        #             phase_err[jj],
        #             omega_err[jj],
        #         )
        #         kk += 1

        # n = self._gen.standard_normal(L) + 1j * self._gen.standard_normal(L)
        # A = 2.0 / self._sqrtN * np.sqrt(cno / T)
        # return np.reshape((A * I) + (Sigma @ n), shape=(-1, 3))

        I = np.zeros((3, 4), order="F", dtype=np.complex128)
        for jj in range(chip.size):
            for ii in range(3):
                I[ii, jj] = self.CalculateI(
                    T,
                    chip_err[jj] + self._tap_space[ii],
                    chip_rate_err[jj],
                    phase_err[jj],
                    omega_err[jj],
                )

        n = self._gen.standard_normal(size=(3, 4)) + 1j * self._gen.standard_normal(size=(3, 4))
        A = 2.0 / self._sqrtN * np.sqrt(cno / T)
        return (A * I + self._Sigma @ n).T

    def CalculateJ(self, T, chip_err, chip_rate_err, phase_err, omega_err):
        phase_exp = np.exp(1j * phase_err)
        if np.abs(omega_err) < 1e-8:
            if (chip_err + (0.5 * T * chip_rate_err)) >= 0.0:
                return phase_exp * T * (1.0 - chip_err - (0.5 * chip_rate_err * T))
            else:
                return phase_exp * T * (1.0 + chip_err + (0.5 * chip_rate_err * T))

        else:
            s = 1j * omega_err
            if (chip_err + (0.5 * T * chip_rate_err)) >= 0.0:
                J = np.exp(s * T) * (s * (chip_err + (chip_rate_err * T) - 1.0) - chip_rate_err)
                J -= s * (chip_err - 1.0) - chip_rate_err
                J *= phase_exp / (omega_err * omega_err)
            else:
                J = np.exp(s * T) * (s * (-chip_err - (chip_rate_err * T) - 1.0) + chip_rate_err)
                J -= s * (-chip_err - 1.0) + chip_rate_err
                J *= phase_exp / (omega_err * omega_err)

        return J

    def CalculateI(self, T, chip_err, chip_rate_err, phase_err, omega_err):
        if chip_rate_err < 1e-8:
            if np.abs(chip_err) >= 1.0:
                return np.complex128(0.0, 0.0)
            low_lim = 0.0
            up_lim = T
            t0 = -1.0
        else:
            low_lim = (-1.0 - chip_err) / chip_rate_err
            up_lim = (1.0 - chip_err) / chip_rate_err
            if low_lim > up_lim:
                tmp = up_lim
                up_lim = low_lim
                low_lim = tmp

            if (low_lim > T) or (up_lim < 0.0):
                return 0.0

            low_lim = np.max([low_lim, 0.0])
            up_lim = np.min([up_lim, T])
            t0 = -chip_err / chip_rate_err

        chip_offset = chip_err + (chip_rate_err * low_lim)
        phase_offset = phase_err + (omega_err * low_lim)
        if (t0 > low_lim) and (t0 < up_lim):
            I = self.CalculateJ(t0 - low_lim, chip_offset, chip_rate_err, phase_offset, omega_err)
            chip_offset = chip_err + (chip_rate_err * t0)
            phase_offset = phase_err + (omega_err * t0)
            I += self.CalculateJ(up_lim - t0, chip_offset, chip_rate_err, phase_offset, omega_err)
        else:
            I = self.CalculateJ(
                up_lim - low_lim, chip_offset, chip_rate_err, phase_offset, omega_err
            )

        return I

    @staticmethod
    def __gen_svd(Q: np.ndarray[np.double]):
        U, S, _ = np.linalg.svd(Q)
        return U @ np.diag(np.sqrt(S))

    @staticmethod
    def __gen_eigh(Q: np.ndarray[np.double]):
        d, W = np.linalg.eigh(Q)
        return W @ np.diag(np.sqrt(d))

    @staticmethod
    def __gen_cholesky(Q: np.ndarray[np.double]):
        return np.linalg.cholesky(Q)


#! ---------------------------------------------------------------------------------------------- !#


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(2, 1)
    for kk in range(3):
        if kk == 0:
            method = "cholesky"
        elif kk == 1:
            method = "eigh"
        elif kk == 2:
            method = "svd"

        sim = ClockModel(method=method, seed=1)
        bias, drift = np.zeros(5000), np.zeros(5000)
        T = 0.01
        for ii in range(5000):
            bias[ii], drift[ii] = sim.sim(T)

        ax[0].plot(bias, label=method)
        ax[1].plot(drift)

    sim = ClockModel(method="svd", seed=1)
    bias, drift = sim.gen(T, 5000)
    ax[0].plot(bias, ":", label="cumsum")
    ax[1].plot(drift, ":")

    ax[0].legend()
    plt.show()
