import asyncio
import json

from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Context, StopEvent, Workflow, step
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

from app.core.model import llm as default_llm
from app.core.prompts import PROMPT
from app.core.schemas import (
    AnalysisStartEvent,
    Insight,
    InsightEvent,
    ParallelAnalysisDoneEvent,
)

MCP_URL = "http://localhost:8001/sse"


def _to_dict(tool_output) -> dict:
    raw = getattr(tool_output, "raw_output", tool_output)

    structured = getattr(raw, "structuredContent", None)
    if isinstance(structured, dict):
        if set(structured) == {"result"} and isinstance(structured["result"], dict):
            return structured["result"]
        return structured

    content = getattr(raw, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text:
            return json.loads(text)

    if isinstance(raw, dict):
        return raw

    raise ValueError(f"Cannot convert tool output to dict: {type(raw)}")


class AudioAnalysisWorkflow(Workflow):
    def __init__(self, mcp_url: str = MCP_URL, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.mcp_url = mcp_url
        self.llm = llm or default_llm
        self._tools = None

    async def _get_tools(self) -> dict:
        if self._tools is None:
            spec = McpToolSpec(client=BasicMCPClient(self.mcp_url))
            tool_list = await spec.to_tool_list_async()
            self._tools = {t.metadata.name: t for t in tool_list}
        return self._tools

    @step
    async def run_parallel_analysis(
        self, ctx: Context, ev: AnalysisStartEvent
    ) -> ParallelAnalysisDoneEvent:
        tools = await self._get_tools()
        fp = ev.file_path

        (
            metadata,
            silence,
            clipping,
            low_volume,
            snr,
            features,
            glitches,
        ) = await asyncio.gather(
            tools["extract_metadata_tool"].acall(file_path=fp),
            tools["detect_silence_tool"].acall(file_path=fp),
            tools["detect_clipping_tool"].acall(file_path=fp),
            tools["detect_low_volume_tool"].acall(file_path=fp),
            tools["estimate_snr_tool"].acall(file_path=fp),
            tools["extract_features_tool"].acall(file_path=fp),
            tools["detect_glitches_tool"].acall(file_path=fp),
        )

        return ParallelAnalysisDoneEvent(
            file_path=fp,
            metadata=_to_dict(metadata),
            silence=_to_dict(silence),
            clipping=_to_dict(clipping),
            low_volume=_to_dict(low_volume),
            snr=_to_dict(snr),
            features=_to_dict(features),
            glitches=_to_dict(glitches),
        )

    @step
    async def generate_insight(
        self, ctx: Context, ev: ParallelAnalysisDoneEvent
    ) -> InsightEvent:
        metrics = {
            "metadata": ev.metadata,
            "silence": ev.silence,
            "low_volume": ev.low_volume,
            "clipping": ev.clipping,
            "snr": ev.snr,
            "features": ev.features,
            "glitches": ev.glitches,
        }
        insight = await self.llm.astructured_predict(
            Insight, PromptTemplate(PROMPT), report=json.dumps(metrics, indent=2)
        )

        return InsightEvent(
            file_path=ev.file_path,
            metrics=metrics,
            insight=insight.model_dump(),
        )

    @step
    async def build_final_report(self, ctx: Context, ev: InsightEvent) -> StopEvent:
        return StopEvent(
            result={
                "file_path": ev.file_path,
                "metrics": ev.metrics,
                "insight": ev.insight,
            }
        )


async def analyze(file_path: str, mcp_url: str = MCP_URL) -> dict:
    wf = AudioAnalysisWorkflow(mcp_url=mcp_url, timeout=300)
    return await wf.run(file_path=file_path)


async def analyze_batch(file_paths: list[str], mcp_url: str = MCP_URL) -> list[dict]:
    return await asyncio.gather(*(analyze(fp, mcp_url) for fp in file_paths))
