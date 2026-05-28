"""
explainability/__init__.py
"""
from explainability.attention_rollout import (
    compute_attention_rollout,
    rollout_to_heatmap,
    visualize_rollout,
    explain_prediction,
)

__all__ = [
    "compute_attention_rollout",
    "rollout_to_heatmap",
    "visualize_rollout",
    "explain_prediction",
]
