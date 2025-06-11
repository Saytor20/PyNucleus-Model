from pathlib import Path
from typing import List
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class FeedComponent(BaseModel):
    name: str
    mole_fraction: float
    mass_flow_kgh: float

class FeedConditions(BaseModel):
    temperature_c: float
    pressure_kpa: float
    total_flow_kmol_h: float
    components: List[FeedComponent]

class OperatingConditions(BaseModel):
    reflux_ratio: float | None = None
    residence_time_min: float | None = None

class SimulationSettings(BaseModel):
    simulation_name: str
    feed: FeedConditions
    operating: OperatingConditions

class RAGSettings(BaseModel):
    top_k: int = 5
    similarity_threshold: float = 0.3

class LLMSettings(BaseModel):
    summary_length: int = 300

class AppSettings(BaseSettings):
    simulations: List[SimulationSettings]
    rag: RAGSettings = RAGSettings()
    llm: LLMSettings = LLMSettings()
    model_config = {"extra": "forbid"}

def load_settings(path: Path) -> "AppSettings":
    return AppSettings.model_validate_json(path.read_text()) 