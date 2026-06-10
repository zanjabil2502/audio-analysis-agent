import librosa
import numpy as np
from scipy.signal import medfilt

from .utils import _load_audio


async def detect_silence(
    file_path: str, top_db: int = 40, frame_length: int = 2048, hop_length: int = 512
) -> dict:
    y, sr = _load_audio(file_path)
    audio_duration = librosa.get_duration(y=y, sr=sr)
    non_silent_intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    silent_intervals = []
    last_end = 0
    total_silent_duration = 0

    for start, end in non_silent_intervals:
        start_sec = start / sr
        if start_sec > last_end:
            duration = start_sec - last_end
            silent_intervals.append(
                {
                    "start_sec": round(last_end, 3),
                    "end_sec": round(start_sec, 3),
                    "duration_sec": round(duration, 3),
                }
            )
            total_silent_duration += duration
        last_end = end / sr

    if last_end < audio_duration:
        duration = audio_duration - last_end
        silent_intervals.append(
            {
                "start_sec": round(last_end, 3),
                "end_sec": round(audio_duration, 3),
                "duration_sec": round(duration, 3),
            }
        )
        total_silent_duration += duration

    return {
        "silence_segments": silent_intervals,
        "total_silence_duration_sec": round(total_silent_duration, 3),
        "silence_ratio": round(total_silent_duration / audio_duration, 3),
    }


async def detect_low_volume(file_path: str, threshold_db: float = -30.0) -> dict:

    y, sr = _load_audio(file_path)
    rms = librosa.feature.rms(y=y)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    mean_db = np.mean(rms_db)

    return {
        "mean_volume_db": round(float(mean_db), 2),
        "low_volume_detected": bool(mean_db <= threshold_db),
    }


async def detect_clipping(file_path: str, threshold: float = 0.99) -> dict:
    y, sr = _load_audio(file_path)
    max_amp = np.max(np.abs(y))
    clipping_samples = np.sum(np.abs(y) >= threshold)
    clipping_ratio = clipping_samples / len(y)

    return {
        "peak_amplitude": round(float(max_amp), 4),
        "clipping_detected": bool(max_amp >= threshold),
        "clipping_ratio": round(float(clipping_ratio), 6),
    }


async def estimate_snr(file_path: str) -> dict:
    y, sr = _load_audio(file_path)

    non_silent_idx = librosa.effects.split(y, top_db=30)
    signal_parts = [y[start:end] for start, end in non_silent_idx]
    if not signal_parts:
        return {"snr_db": 0, "error": "No signal detected"}

    signal_energy = np.mean([np.mean(part**2) for part in signal_parts])
    noise_energy = np.mean(y**2) - signal_energy
    noise_energy = max(noise_energy, 1e-10)

    snr_db = 10 * np.log10(signal_energy / noise_energy)

    return {
        "snr_db": round(float(snr_db), 2),
        "quality_category": "Excellent"
        if snr_db > 30
        else "Good"
        if snr_db > 20
        else "Poor",
    }


def _merge_times(times, max_gap_sec):
    if len(times) == 0:
        return []
    segments = []
    start = prev = float(times[0])
    for t in times[1:]:
        t = float(t)
        if t - prev <= max_gap_sec:
            prev = t
        else:
            segments.append((start, prev))
            start = prev = t
    segments.append((start, prev))
    return segments


def _rolling_energy(x, win):
    c = np.cumsum(np.concatenate(([0.0], x.astype(np.float64) ** 2)))
    return np.maximum(c[win:] - c[:-win], 0.0)


async def detect_glitches(
    file_path: str,
    click_sigma: float = 12.0,
    click_max_ms: float = 3.0,
    dropout_min_ms: float = 2.0,
    dropout_max_ms: float = 120.0,
    stutter_ratio: float = 0.1,
    max_events: int = 100,
) -> dict:
    y, sr = _load_audio(file_path)
    peak = float(np.max(np.abs(y))) + 1e-9
    events: list[dict] = []

    resid = y - medfilt(y, kernel_size=5)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    score = np.abs(resid) / (1.4826 * mad)
    click_times = np.where(score > click_sigma)[0] / sr
    for start, end in _merge_times(click_times, max_gap_sec=0.005):
        if end - start > click_max_ms / 1000:
            continue
        i0, i1 = int(start * sr), min(int(end * sr) + 1, len(score))
        events.append(
            {
                "type": "CLICK",
                "start_sec": round(start, 3),
                "duration_sec": round(end - start, 3),
                "confidence": round(
                    min(float(score[i0:i1].max()) / (3 * click_sigma), 1.0), 2
                ),
            }
        )

    eps = 1e-4 * peak
    silent = (np.abs(y) < eps).astype(np.int8)
    d = np.diff(np.concatenate(([0], silent, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    min_len = int(dropout_min_ms * sr / 1000)
    max_len = int(dropout_max_ms * sr / 1000)
    guard = int(0.005 * sr)
    for s, e in zip(starts, ends):
        if not (min_len <= e - s <= max_len):
            continue
        left = y[max(0, s - guard) : s]
        right = y[e : e + guard]
        if (
            left.size
            and right.size
            and np.max(np.abs(left)) > 5 * eps
            and np.max(np.abs(right)) > 5 * eps
        ):
            events.append(
                {
                    "type": "DROPOUT",
                    "start_sec": round(float(s / sr), 3),
                    "duration_sec": round(float((e - s) / sr), 3),
                }
            )

    win = int(0.02 * sr)
    sig_e = _rolling_energy(y, win)
    active = sig_e > win * (0.02 * peak) ** 2
    best_ratio = np.full(len(sig_e), np.inf)
    best_lag = np.zeros(len(sig_e), dtype=int)
    for lag in range(int(0.01 * sr), int(0.06 * sr), int(0.005 * sr)):
        e = y[lag:] - y[:-lag]
        num = _rolling_energy(e, win)
        m = len(num)
        ratio = np.sqrt(num / (sig_e[:m] + 1e-9))
        upd = ratio < best_ratio[:m]
        best_ratio[:m][upd] = ratio[upd]
        best_lag[:m][upd] = lag
    stutter_times = np.where((best_ratio < stutter_ratio) & active)[0] / sr
    for start, end in _merge_times(stutter_times, max_gap_sec=0.03):
        if end - start < win / sr:
            continue
        events.append(
            {
                "type": "STUTTER",
                "start_sec": round(start, 3),
                "duration_sec": round(end - start, 3),
                "repeat_lag_ms": round(float(best_lag[int(start * sr)] / sr * 1000), 1),
            }
        )

    events.sort(key=lambda ev: ev["start_sec"])
    counts = {"CLICK": 0, "DROPOUT": 0, "STUTTER": 0}
    for ev in events:
        counts[ev["type"]] += 1

    return {
        "glitch_count": len(events),
        "click_count": counts["CLICK"],
        "dropout_count": counts["DROPOUT"],
        "stutter_count": counts["STUTTER"],
        "events": events[:max_events],
        "events_truncated": len(events) > max_events,
    }
