from netdef_slim.core.base_struct import BaseStruct
from netdef_slim.core.register import register_class
import numpy as np

nothing = None


class _DataStruct(BaseStruct):
    def __init__(self):
        super().__init__()

    def is_data(self, member):
        return not self.is_struct(member)

    def is_struct(self, member):
        return isinstance(self._members[member], _DataStruct)

    def make_struct(self, member):
        if not member in self._members: self[member] = _DataStruct()

    def to_string(self, indent):
        s = ''
        for name, member in self._members.items():
            if isinstance(member, np.ndarray):
                s += '%s np.array: %s %s\n' % (' '*indent, name, member.shape)
            elif isinstance(member, _DataStruct):
                s += '%s struct: %s\n' % (' '*indent, name)
                s += member.to_string(indent+2)
            else:
                s += '%s %s = %s\n' % (' '*indent, name, str(member))
        if s == '': s = "%s (empty)\n" % (' '*indent)
        return s


register_class('DataStruct', _DataStruct)


