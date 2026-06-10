from fastmcp import FastMCP

from audio_mcp.registry import (
    register_detect_audio,
    register_extract_features,
    register_extract_metadata,
)

mcp = FastMCP(
    name="Audio Analysis MCP Server",
    instructions="Use this server to analyze audio files for quality assessment.",
)

register_detect_audio(mcp)
register_extract_features(mcp)
register_extract_metadata(mcp)

if __name__ == "__main__":
    mcp.run()
