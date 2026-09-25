"""Load trusted local checkpoints after the GEVD package reorganization."""
import pickle


_MODULES = {
    "trainer": "gevd.training.base",
    "utils.replay_buffer": "gevd.training.replay",
    "utils.gauge_env": "gevd.environments.multi_robot",
    "utils.retrospective_utility": "gevd.training.retrospective_utility",
    "utils.event_aligned_credit": "gevd.training.event_credit",
    "utils.learned_event_credit": "gevd.training.learned_event_credit",
    "utils.paired_return_credit": "gevd.training.paired_return_credit",
}


class Unpickler(pickle.Unpickler):
    """Translate known historical module names without loading the old source tree."""

    def find_class(self, module, name):
        return super().find_class(_MODULES.get(module, module), name)


# torch.load accepts a pickle-module interface; serialization remains standard pickle.
Pickler = pickle.Pickler
load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps
