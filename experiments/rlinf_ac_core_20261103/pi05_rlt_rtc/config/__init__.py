"""Design-config parsing and strict validation."""

from pi05_rlt_rtc.config.design_config import (
    DESIGN_SCHEMA_VERSION,
    AssetsConfig,
    DesignConfig,
    EnvironmentConfig,
    EvaluationConfig,
    FeatureModelConfig,
    IntegrationConfig,
    LearningConfig,
    RTCConfig,
    RuntimeConfig,
    SourceConfig,
    load_design_config,
)

__all__ = [
    "DESIGN_SCHEMA_VERSION",
    "AssetsConfig",
    "DesignConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "FeatureModelConfig",
    "IntegrationConfig",
    "LearningConfig",
    "RTCConfig",
    "RuntimeConfig",
    "SourceConfig",
    "load_design_config",
]
