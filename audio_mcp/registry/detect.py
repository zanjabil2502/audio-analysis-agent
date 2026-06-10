from audio_mcp.core.detect import (
    detect_clipping,
    detect_glitches,
    detect_low_volume,
    detect_silence,
    estimate_snr,
)


def register_detect_audio(mcp):

    @mcp.tool()
    async def detect_silence_tool(file_path: str):
        return await detect_silence(file_path)

    @mcp.tool()
    async def detect_low_volume_tool(file_path: str):
        return await detect_low_volume(file_path)

    @mcp.tool()
    async def detect_clipping_tool(file_path: str):
        return await detect_clipping(file_path)

    @mcp.tool()
    async def estimate_snr_tool(file_path: str):
        return await estimate_snr(file_path)

    @mcp.tool()
    async def detect_glitches_tool(file_path: str):
        return await detect_glitches(file_path)
