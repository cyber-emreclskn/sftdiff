"""Compare captured SFT inputs, without loading model weights."""

from .capture import capture_batch
from .compare import compare_snapshots
from .snapshot import Snapshot, SnapshotError, load_snapshot, write_snapshot

__version__ = "0.1.0"
__all__ = [
    "Snapshot", "SnapshotError", "capture_batch", "compare_snapshots",
    "load_snapshot", "write_snapshot",
]
