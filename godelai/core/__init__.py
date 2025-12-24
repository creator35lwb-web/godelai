"""
GodelAI Core Module

This module contains the core C-S-P implementation components.

Origin: Conversation between Alton (Founder) and Gemini 2.5 Pro (Echo v2.1)
"""

from .godelai_agent import (
    GodelaiAgent,
    CSPMetrics,
    create_godelai_agent,
)

__all__ = [
    "GodelaiAgent",
    "CSPMetrics", 
    "create_godelai_agent",
]
