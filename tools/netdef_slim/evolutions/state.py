import re, os
import netdef_slim as nd

class _State:
    def __init__(self, id):
        id_re_match = re.compile('([^:]+):([0-9]+)').match(id)
        self._id = id
        self._evo_name = id_re_match.group(1)
        self._iter = int(id_re_match.group(2))

    def iter(self): return self._iter
    def id(self): return self._id
    def evo_name(self): return self._evo_name

    def evo_index(self):
        return nd.evo_manager.get_evolution(self._evo_name).index()

    def __lt__(self, other):
        return (self.evo_index() < other.evo_index()) or (
                    self.evo_index() == other.evo_index() and self.iter() < other.iter())

    def __le__(self, other):
        return (self.evo_index() < other.evo_index()) or (
                    self.evo_index() == other.evo_index() and self.iter() <= other.iter())

    def __gt__(self, other):
        return (self.evo_index() > other.evo_index()) or (
                    self.evo_index() == other.evo_index() and self.iter() > other.iter())

    def __ge__(self, other):
        return (self.evo_index() > other.evo_index()) or (
                    self.evo_index() == other.evo_index() and self.iter() >= other.iter())

    def __eq__(self, other):
        return self.evo_index() == other.evo_index() and self.iter() == other.iter()

    def __ne__(self, other):
        return self.evo_index() != other.evo_index() or self.iter() != other.iter()

    def folder(self):
        return os.path.join(nd.evo_manager.training_dir(), self.evo_name())

    def __str__(self):
        return self._evo_name+':'+str(self._iter)

