from netdef_slim.core.base_struct import BaseStruct
from netdef_slim.core.register import register_class

nothing = None

class _Struct(BaseStruct):
    def __init__(self):
        super().__init__()

    def is_data(self, member):
        return not self.is_struct(member)

    def is_struct(self, member):
        return isinstance(self._members[member], _Struct)

    def to_string(self, indent):
        s = str(self._members)
        return s

    def make_sibling(self, other):
        pass

register_class('Struct', _Struct)