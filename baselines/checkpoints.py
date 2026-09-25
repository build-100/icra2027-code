"""Load trusted baseline checkpoints saved before the package reorganization."""
import pickle
from types import ModuleType

import torch


class _BaselineUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'utils.replay_buffer':
            module = 'baselines.replay'
        return super().find_class(module, name)


_PICKLE = ModuleType('baseline_checkpoint_pickle')
_PICKLE.Unpickler = _BaselineUnpickler
_PICKLE.Pickler = pickle.Pickler
_PICKLE.load = pickle.load
_PICKLE.loads = pickle.loads
_PICKLE.dump = pickle.dump
_PICKLE.dumps = pickle.dumps


def load_checkpoint(path, map_location='cpu'):
    """Read a trusted checkpoint and remap the historical replay tuple module.

    This preserves network keys, optimizer state and replay values. As with
    ordinary PyTorch optimizer checkpoints, only load files you trust.
    """
    return torch.load(path, map_location=map_location, pickle_module=_PICKLE, weights_only=False)
