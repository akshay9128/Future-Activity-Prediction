"""Streamlit web application package for activity prediction."""

from src.app.streamlit_app import (
    ensure_web_compatible_video,
    get_cached_predictor,
    get_sample_videos_from_manifest,
    main,
)

__all__ = [
    "ensure_web_compatible_video",
    "get_cached_predictor",
    "get_sample_videos_from_manifest",
    "main",
]
