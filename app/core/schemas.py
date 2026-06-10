from llama_index.core.workflow import Event, StartEvent
from pydantic import BaseModel


class IssueExplained(BaseModel):
    issue: str
    explanation: str
    impact: str


class Insight(BaseModel):
    summary: str
    issues_explained: list[IssueExplained]
    asr_impact: str
    recommendations: list[str]


class AnalysisStartEvent(StartEvent):
    file_path: str


class ParallelAnalysisDoneEvent(Event):
    file_path: str
    metadata: dict
    silence: dict
    clipping: dict
    low_volume: dict
    snr: dict
    features: dict
    glitches: dict


class InsightEvent(Event):
    file_path: str
    metrics: dict
    insight: dict
