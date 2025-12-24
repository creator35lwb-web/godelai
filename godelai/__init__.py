"""
GodelAI - An open-source small language model built on the C-S-P framework
==========================================================================

C-S-P Model: Compression → State → Propagation

Core Thesis:
    "Wisdom is not an entity, but a process structure that is 
    continuously executed and inherited."

Alignment Principle:
    The system can optimize any goal, but must preserve the 
    transmissibility of "the ability to modify goals."
"""

__version__ = "0.1.0"
__author__ = "Alton & Godel (Manus AI)"

from godelai.reg.csp_regularizer import (
    CSPRegularizer,
    CSPTrainerCallback,
    csp_state,
    is_alive,
)

__all__ = [
    "CSPRegularizer",
    "CSPTrainerCallback", 
    "csp_state",
    "is_alive",
    "__version__",
]
