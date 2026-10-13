"""
Streamlit application module.

Provides direct access and compatibility alias for the interactive dashboard.
"""

from src.app.streamlit_app import (
    ensure_web_compatible_video,
    get_cached_predictor,
    get_sample_videos_from_manifest,
    main,
    render_attention_chart,
)

__all__ = [
    "ensure_web_compatible_video",
    "get_cached_predictor",
    "get_sample_videos_from_manifest",
    "main",
    "render_attention_chart",
]

if __name__ == "__main__":
    main()
