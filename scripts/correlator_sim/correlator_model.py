import numpy as np


class CorrelatorSim:

    def __init__(self, tap_space, n_sv, n_segments):
        self.SetTapSpacing(tap_space)
        self.M_ = n_sv
        self.sqrtN_ = np.sqrt(n_segments)
        self.I_ = np.zeros((3, 3))
        self.T_accum_ = 0.0
        self.k_ = 1.0

    def SetTapSpacing(self, tap_space):
        self.taps_ = np.array([-tap_space, 0.0, tap_space])
        Iep = self.CalculateI(1.0, self.taps_[0], 0.0, 0.0, 0.0)
        Ipp = self.CalculateI(1.0, self.taps_[1], 0.0, 0.0, 0.0)
        Ipl = self.CalculateI(1.0, self.taps_[2], 0.0, 0.0, 0.0)
        Iel = self.CalculateI(1.0, self.taps_[0] - self.taps_[2], 0.0, 0.0, 0.0)
        SigmaSq = np.array([Ipp, np.conj(Iep), Iel, Iep, Ipp, np.conj(Ipl), np.conj(Iel), Ipl, Ipp])
        self.Sigma_ = np.linalg.cholesky(SigmaSq)

    def NextSample(
        self,
        T,
        true_cno,
        true_chip_phase,
        true_chip_rate,
        true_carrier_phase,
        true_carrier_freq,
        est_chip_phase,
        est_chip_rate,
        est_carrier_phase,
        est_carrier_freq,
    ):
        # accumulate error in a piecewise manner
        chip_err = true_chip_phase - est_chip_phase
        chip_rate_err = true_chip_rate - est_chip_rate
        phase_err = true_carrier_phase - est_carrier_phase
        omega_err = true_carrier_freq - est_carrier_freq
        # print(g"c_err = {chip_err.transpose()}")
        # print(g"fc_err = {chip_rate_err.transpose()}")
        # print(g"p_err = {phase_err.transpose()}")
        # print(g"w_err = {omega_err.transpose()}")
        for ii in range(self.M_):
            for jj in range(3):
                self.I_[jj, ii] += self.CalculateI(
                    T,
                    chip_err[ii] + self.taps_[jj],
                    chip_rate_err[ii],
                    phase_err[ii],
                    omega_err[ii],
                )

        # average the cno over the piecewise integral
        self.cno_avg_ += (1.0 / self.k_) * (true_cno - self.cno_avg_)
        self.k_ += 1.0
        self.T_accum_ += T

    def GetCorrelators(self):
        # calculate correlators
        R = np.zeros((3, self.M_), order="F")
        for ii in range(self.M_):
            n = np.random.randn(3) + 1j * np.random.randn(3)
            A = 2.0 / self.sqrtN_ * np.sqrt(self.cno_avg_[ii] / self.T_accum_)
            R[:, ii] = (A * self.I_[:, ii]) + (self.Sigma_ * n)

        # reset piecewise integral
        self.I_[:] = 0.0
        self.cno_avg_[:] = 0.0
        self.k_ = 1.0
        self.T_accum_ = 0.0

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


if __name__ == "__main__":
    sim = CorrelatorSim(0.5, 2, 2)
