import json
from pathlib import Path

from .utils import _run_ffmpeg


async def extract_metadata(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        return {"error": "File does not exist"}

    returncode, stdout, stderr = await _run_ffmpeg(
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    )

    if returncode != 0:
        return {"error": stderr}

    raw: dict = json.loads(stdout)
    stream = next(
        (s for s in raw.get("streams", []) if s.get("codec_type") == "audio"), {}
    )

    fmt: dict = raw.get("format", {})

    return {
        "file_name": path.name,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "format": fmt.get("format_name"),
        "duration_seconds": round(float(fmt.get("duration", 0)), 2),
        "bitrate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000, 1),
        "codec": stream.get("codec_name"),
        "sample_rate_hz": int(stream.get("sample_rate", 0)),
        "channels": stream.get("channels"),
        "channel_layout": stream.get("channel_layout"),
        "bit_depth": stream.get("bits_per_sample"),
        "creation_time": fmt.get("tags", {}).get("creation_time"),
    }
