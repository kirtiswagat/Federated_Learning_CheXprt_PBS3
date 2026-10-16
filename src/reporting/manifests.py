"""Reporting helpers that only consume validated artifact manifests."""

from src.monitoring.artifact_registry import validate_manifest

__all__ = ["validate_manifest"]
