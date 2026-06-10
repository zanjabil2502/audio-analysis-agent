from audio_mcp.core.feature import extract_features


def register_extract_features(mcp):

    @mcp.tool()
    async def extract_features_tool(file_path: str):
        return await extract_features(file_path)
