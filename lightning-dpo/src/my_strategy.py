from logging import WARN
from typing import Any, Dict, List
from lightning.fabric.plugins import ClusterEnvironment
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.strategies import DeepSpeedStrategy
from typing import Optional
from lightning.fabric.utilities.types import _PATH, LRScheduler, ReduceLROnPlateau
from typing_extensions import override

class MyDeepSpeedStrategy(DeepSpeedStrategy):
    @override
    def save_checkpoint(self, checkpoint: Dict, filepath: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: The checkpoint state dictionary
            filepath: write-target file's path
            storage_options: not used for ``DeepSpeedStrategy`` as ``CheckpointIO`` is not used

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in

        """
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)