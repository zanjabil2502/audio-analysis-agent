from pathlib import Path

import librosa
import numpy as np

from .utils import _load_audio, band_energy_ratio


async def extract_spectral(y: list, sr: int, n_fft: int, hop_length: int) -> dict:
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
    )[0]

    spectral_flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length
    )[0]

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    return {
        "centroid_mean_hz": round(float(np.mean(spectral_centroid)), 1),
        "centroid_std_hz": round(float(np.std(spectral_centroid)), 1),
        "bandwidth_mean_hz": round(float(np.mean(spectral_bandwidth)), 1),
        "rolloff_mean_hz": round(float(np.mean(spectral_rolloff)), 1),
        "flatness_mean": round(float(np.mean(spectral_flatness)), 4),
        "zcr_mean": round(float(np.mean(zcr)), 4),
    }


async def extract_band(D, freqs, sr) -> dict:
    speech_band_ratio = band_energy_ratio(D, freqs, 300, 3400)
    sub_bass_ratio = band_energy_ratio(D, freqs, 20, 80)
    hum_50hz_ratio = band_energy_ratio(D, freqs, 49, 51)
    hum_60hz_ratio = band_energy_ratio(D, freqs, 59, 61)
    high_freq_ratio = band_energy_ratio(D, freqs, 8000, sr // 2)

    return {
        "speech_band_ratio": round(speech_band_ratio, 4),
        "sub_bass_ratio": round(sub_bass_ratio, 4),
        "high_freq_ratio": round(high_freq_ratio, 4),
        "hum_50hz_ratio": round(hum_50hz_ratio, 6),
        "hum_60hz_ratio": round(hum_60hz_ratio, 6),
    }


async def extract_features(
    file_path: str,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict:

    if not Path(file_path).exists():
        return {"error": f"File not found: {file_path}"}

    y, sr = _load_audio(file_path)
    duration = librosa.get_duration(y=y, sr=sr)

    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    spectral_features = await extract_spectral(y, sr, n_fft, hop_length)
    band_features = await extract_band(D, freqs, sr)

    return {
        "file_path": file_path,
        "duration_seconds": round(duration, 2),
        "spectral_features": spectral_features,
        "band_energy": band_features,
    }
