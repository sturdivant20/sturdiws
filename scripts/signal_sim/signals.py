import numpy as np
from scipy.signal import butter, cheby1, cheby2, lfilter, decimate, welch
from pathlib import Path


def combine_int16_iq_files(
    sig_file: Path,
    awgn_file: Path,
    out_file: Path,
    chunksize: int = 2**29,  # 0.5 GB,
) -> None:
    """
    Sums two large binary files together while avoiding int16 overflow.
    """
    with open(sig_file, "rb") as fs, open(awgn_file, "rb") as fn, open(out_file, "wb") as fo:
        while True:
            # read chunks
            s_chunk = np.frombuffer(fs.read(chunksize), dtype=np.int16)
            n_chunk = np.frombuffer(fn.read(chunksize), dtype=np.int16)
            if s_chunk.size == 0 and n_chunk.size == 0:
                return

            # pad
            if s_chunk.size < n_chunk.size:
                s_chunk = np.pad(s_chunk, (0, n_chunk.size - s_chunk.size), mode="constant")
            elif n_chunk.size < s_chunk.size:
                n_chunk = np.pad(n_chunk, (0, s_chunk.size - n_chunk.size), mode="constant")

            # sum together (signal is at 80 db and noise is at 74 dB so dividing the signal by 2
            # should properly handle overflow)
            sum_chunk = np.round(s_chunk / 2 + n_chunk).astype(np.int16)

            # write output
            fo.write(sum_chunk.tobytes())
            fo.flush()
    return


def write_int16_noise_file(
    noise_file: Path,
    seconds: float,
    seed: int,
    bandwidth: float,
    sampling_rate: float,
    filter_order: int = 10,
    power_dbm: float = 1,
    gain_db: float = 1,
):
    gen = np.random.default_rng(seed)
    samp_rem = int(seconds * sampling_rate)
    with open(noise_file, "wb") as fn:
        while True:
            num_samp = min(samp_rem, 2**26)
            noise_signal = inflate_signal_power(
                baseband_blwn(gen, bandwidth, sampling_rate, num_samp, filter_order, power_dbm),
                gain_db,
            )
            out = np.vstack(
                (
                    np.round(noise_signal.real).astype(np.int16),
                    np.round(noise_signal.imag).astype(np.int16),
                ),
                dtype=np.int16,
            ).reshape((-1,), order="F")
            fn.write(out.tobytes())
            fn.flush()
            samp_rem -= num_samp
            if samp_rem <= 0:
                break
    return


def downsample_int16_file(
    in_file: Path,
    out_file: Path,
    factor: int = 2,
    chunksize: int = 2**30,  # 1.0 GB,
):
    with open(in_file, "rb") as fs, open(out_file, "wb") as fo:
        while True:
            # read chunks
            s_chunk = np.frombuffer(fs.read(chunksize), dtype=np.int16)

            # downsample with iir filter
            down_chunk = np.vstack(
                (
                    np.round(decimate(s_chunk[0::2], factor, zero_phase=True)).astype(np.int16),
                    np.round(decimate(s_chunk[1::2], factor, zero_phase=True)).astype(np.int16),
                ),
                dtype=np.int16,
            ).reshape((-1,), order="F")

            # write output
            fo.write(down_chunk.tobytes())
            fo.flush()
    return


def baseband_blwn(
    gen: np.random.Generator,
    bandwidth: float,
    sampling_rate: float,
    num_samples: int,
    filter_order: int = 10,
    power_dbm: float = 1,
) -> np.ndarray[np.complex128]:
    """
    Generates complex bandlimited white noise at baseband with a specified power level.

    Args:
        bandwidth (float): The desired bandwidth (Hz).
        sampling_rate (float): The sampling rate (Hz).
        duration (float): The duration (seconds).
        power_dbm (float): The desired power level in dBm.

    Returns:
        numpy.ndarray: The complex baseband bandlimited white noise signal with the desired power.
    """

    # 1. Convert dBm to Watts
    power_watts = 10 ** ((power_dbm - 30) / 10)

    # 2. Calculate the required standard deviation (assuming 1 Ohm impedance)
    std_dev = np.sqrt(power_watts)

    # 3. Generate white Gaussian noise with the calculated standard deviation
    # real_noise = np.random.normal(0, std_dev, size=num_samples)
    # imag_noise = np.random.normal(0, std_dev, size=num_samples)
    real_noise = gen.normal(loc=0, scale=std_dev, size=num_samples)
    imag_noise = gen.normal(loc=0, scale=std_dev, size=num_samples)

    # 4. Design and apply the low-pass filter
    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff = bandwidth / nyquist_freq
    if normalized_cutoff >= 1.0:
        raise ValueError("Bandwidth must be less than the Nyquist frequency.")

    # b, a = butter(filter_order, normalized_cutoff, btype="low", analog=False)
    # b, a = cheby1(filter_order, 0.1, normalized_cutoff, btype="low", analog=False)
    b, a = cheby2(filter_order, 100, normalized_cutoff, btype="low", analog=False)

    bandlimited_real = lfilter(b, a, real_noise)
    bandlimited_imag = lfilter(b, a, imag_noise)
    return bandlimited_real + 1j * bandlimited_imag


def inflate_signal_power(
    signal: np.ndarray[np.complex128], gain_db: float
) -> np.ndarray[np.complex128]:
    """Inflates the power of a signal by a specified amount in dB."""
    amplitude_scale = 10 ** (gain_db / 20)
    return signal * amplitude_scale


#! ---------------------------------------------------------------------------------------------- !#
#! DOWN SAMPLE SIGNAL FILE

# if __name__ == "__main__":
#     # downsample and save skydel signal files
#     from secrets import randbits
#     from multiprocessing import Pool, freeze_support
#     from time import time

#     t0 = time()
#     print("downsampling skydel signal files from 12.5 MHz to 6.25 MHz ... ")
#     # inpath = Path("/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/drone-sim/")
#     # outpath = Path("/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/drone-sim-downsampled/")
#     inpath = Path("/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/ground-sim/")
#     outpath = Path("/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/ground-sim-downsampled/")
#     cnos = np.arange(20, 42, 2, dtype=int)
#     for ii, cno in enumerate(cnos):
#         cno_path = inpath / f"CNo_{cno}_dB"
#         cno_path.mkdir(parents=True, exist_ok=True)
#         out_path = outpath / f"CNo_{cno}_dB"
#         out_path.mkdir(parents=True, exist_ok=True)

#         pool = Pool(processes=4)
#         for jj in range(4):
#             pool.apply_async(
#                 downsample_int16_file,
#                 args=(cno_path / f"Ant-{jj}.bin", out_path / f"Ant-{jj}.bin", 2, 2**30),
#             )
#         pool.close()
#         pool.join()
#     print(f"Finished processing in {time() - t0} seconds.")


#! ---------------------------------------------------------------------------------------------- !#
#! GENERATE NOISE FILES


# if __name__ == "__main__":
#     # generate 30 random noise files at 6.25 MHz
#     from secrets import randbits
#     from multiprocessing import Pool, freeze_support
#     from time import time

#     t0 = time()
#     print("generating noise files at 6.25 MHz and bandwidth 6.1 MHz ... ")
#     freeze_support()
#     # unique_seeds = [randbits(128) for _ in range(30)]
#     seconds = 115.66  # 465.05
#     mypath = Path("/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/drone-sim-downsampled/noise")
#     # mypath = Path("/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/ground-sim-downsampled/noise")
#     mypath.mkdir(parents=True, exist_ok=True)
#     pool = Pool(processes=6)
#     for jj in range(18, 30):
#         for kk in range(4):
#             file = mypath / f"noise-{jj}-{kk}.bin"
#             seed = randbits(128)
#             gen = np.random.default_rng(seed=seed)
#             # write_int16_noise_file(file, seconds, gen, 3.0e6, 6.25e6, 13, 26, 76)
#             pool.apply_async(
#                 write_int16_noise_file,
#                 args=(file, seconds, seed, 3.05e6, 6.25e6, 12, 26.5, 76),
#             )
#     pool.close()
#     pool.join()
#     print(f"Finished processing in {time() - t0} seconds.")

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     noise_file = Path(
#         "/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/ground-sim-downsampled/noise/noise-0.bin"
#     )
#     with open(noise_file, "rb") as f:
#         s = np.fromfile(f, dtype=np.int16, count=25000)
#         i = s[0::2]
#         q = s[1::2]

#     plt.figure()
#     plt.plot(i)
#     plt.plot(q)
#     plt.show()


#! ---------------------------------------------------------------------------------------------- !#
#! MERGE FILES


if __name__ == "__main__":
    from time import time
    from multiprocessing import Pool, freeze_support

    t0 = time()
    print("combining signal and noise files ... ")
    freeze_support()

    # combine signal files
    noise_folder = Path(
        "/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/drone-sim-downsampled/noise/"
    )
    signal_folder = Path(
        "/media/daniel/Sturdivant/Thesis-Data/Skydel-Output/drone-sim-downsampled/CNo_40_dB/"
    )
    out_folder = Path("./data")

    pool = Pool(processes=4)
    for ii in range(4):
        noise_file = noise_folder / f"noise-0-{ii}.bin"
        sig_file = signal_folder / f"Ant-{ii}.bin"
        out_file = out_folder / f"SigSim-{ii}.bin"
        pool.apply_async(combine_int16_iq_files, args=(sig_file, noise_file, out_file, 2**29))
    pool.close()
    pool.join()
    print(f"Finished processing in {time() - t0} seconds.")


#! ---------------------------------------------------------------------------------------------- !#
#! EXAMPLE/TEST


# if __name__ == "__main__":
#     desired_bandwidth = 6.1e6  # 12 MHz
#     sampling_rate = 12.5e6  # 12.5 MHz
#     noise_duration = int(12.5e6 * 1)  # seconds
#     desired_power_dbm = 27  # dBm
#     power_inflation_db = 76  # dB

#     # noise generation with seed
#     gen = np.random.default_rng(seed=0)

#     # generate original signal
#     noise_signal = inflate_signal_power(
#         baseband_blwn(gen, desired_bandwidth, sampling_rate, noise_duration, 12, desired_power_dbm),
#         power_inflation_db,
#     )
#     print(f"Generated bandlimited noise with a target power of {desired_power_dbm} dBm.")
#     print(f"Inflating output power by {power_inflation_db} dB.")
#     print(
#         f"Estimated power of the generated signal: {10 * np.log10(np.mean(np.abs(noise_signal) ** 2) * 1000):.2f} dBm"
#     )

#     # generate original signal at half sampling rate
#     noise_signal_low_fs = inflate_signal_power(
#         baseband_blwn(
#             gen, desired_bandwidth / 2, sampling_rate / 2, noise_duration, 12, desired_power_dbm - 3
#         ),
#         power_inflation_db,
#     )
#     print(f"Generated bandlimited noise with a target power of {desired_power_dbm} dBm.")
#     print(f"Inflating output power by {power_inflation_db} dB.")
#     print(
#         f"Estimated power of the generated signal: {10 * np.log10(np.mean(np.abs(noise_signal) ** 2) * 1000):.2f} dBm"
#     )

#     # calculate PSD
#     frequencies1, psd1 = welch(noise_signal, sampling_rate, nperseg=2**15, return_onesided=False)
#     frequencies2, psd2 = welch(
#         noise_signal_low_fs, sampling_rate // 2, nperseg=2**15, return_onesided=False
#     )
#     frequencies1 = np.fft.fftshift(frequencies1)
#     psd1 = np.fft.fftshift(psd1)
#     frequencies2 = np.fft.fftshift(frequencies2)
#     psd2 = np.fft.fftshift(psd2)

#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(10, 6))
#     plt.plot(frequencies1, 10 * np.log10(psd1 * 1000), label="High Sampling Rate")
#     plt.plot(frequencies2, 10 * np.log10(psd2 * 1000), label="Low Sampling Rate")
#     plt.legend()
#     plt.title("Power Spectral Density of Baseband Bandlimited White Noise")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("PSD (dBm/Hz)")
#     plt.grid(True)
#     plt.xlim(-desired_bandwidth * 1.25, desired_bandwidth * 1.25)

#     plt.show()
