from audio_mcp.core.metadata import extract_metadata


def register_extract_metadata(mcp):

    @mcp.tool()
    async def extract_metadata_tool(file_path: str):
        return await extract_metadata(file_path)
