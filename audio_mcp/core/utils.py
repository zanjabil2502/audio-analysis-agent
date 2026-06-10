import asyncio

import librosa
import numpy as np


async def _run_ffmpeg(*args) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()


def _load_audio(file_path: str, sr=None):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr


def band_energy_ratio(D, freqs, low_hz, high_hz):
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not band_mask.any():
        return 0.0
    band_energy = np.mean(D[band_mask, :] ** 2)
    total_energy = np.mean(D**2) + 1e-10
    return float(band_energy / total_energy)
