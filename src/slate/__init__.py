"""SLATE: exactly erasable parametric memory for personal agents.

A ledger of facts is the source of truth; consolidation compiles it into small, sharded, staged adapters; a deletion
request is erased by retraining only the shards whose inputs change, from their checkpoints, and the new state's hash
is a receipt that a replay from scratch reproduces bit for bit.
"""
from .ledger import Fact, Ledger
from .memory import Receipt, SlateMemory
from .planner import Plan
from .train import TrainConfig

__all__ = ["Fact", "Ledger", "Plan", "Receipt", "SlateMemory", "TrainConfig"]
__version__ = "0.1.0"
