#!/usr/bin/env python3

import netdef_slim as nd
from collections import OrderedDict
import re
from copy import copy

rx_int = re.compile('^[0-9]+$')
re_name_idx = re.compile('^(.*?)\\[(\d+)\\]$')

class BaseStruct(object):

    def __init__(self):
        super(BaseStruct, self).__setattr__('_members', OrderedDict())

    def __getattr__(self, index):
        return self._members[index]

    def __setattr__(self, index, value):
        self._members[index] = value

    def __iter__(self):
        return iter(list(self._members.keys()))

    def iteritems(self):
        return iter(self._members.items())

    def keys(self):
        return self._members.keys()

    def values(self):
        return list(self._members.values())

    def make_struct(self, member):
        if not member in self._members: self[member] = nd.Struct()

    def __iter__(self):
        return iter(list(self.keys()))

    def __getitem__(self, index):
        return self.__getattr__(index)

    def __setitem__(self, index, value):
        return self.__setattr__(index, value)

    def to_string(self, indent):
        raise NotImplementedError

    def __str__(self):
        return self.to_string(2)[:-1]

    def set(self, path, value):
        parts = path.split('.')

        def extract_index(part):
            m = re_name_idx.match(part)
            if m:
                return m.group(1), int(m.group(2))
            return part, None

        struct = self
        for part in parts[:-1]:
            part, index = extract_index(part)

            struct.make_struct(part)
            struct = struct[part]

            if index is not None:
                struct.make_struct(index)
                struct = struct[index]

        part, index = extract_index(parts[-1])

        if index is not None:
            struct.make_struct(part)
            struct = struct[part]

            struct[index] = value

        else:
            struct[part] = value

    def _member_to_str(self, value):
        if rx_int.match(str(value)):
            return '[%s]' % int(value)
        return str(value)

    def get_list(self, map_aux=False):
        list = OrderedDict()

        for member in self._members:
            member_name = self._member_to_str(member)

            if self.is_data(member):
                list[member] = self[member]

            elif self.is_struct(member):
                sub_struct = self[member]
                sub_list = sub_struct.get_list()
                for key, value in sub_list.items():
                    if key.startswith('['): joined = member_name + key
                    else: joined = member_name + '.' + key
                    list[joined] = value

            elif map_aux:
                list[member] = self[member]

        return list

    def get(self, full_name):
        for member in self._members:
            member_name = self._member_to_str(member)
            nd.log("mem: {} ".format(member_name))

            if self.is_data(member) and member_name==full_name:
                return self[member]

            elif self.is_struct(member) and full_name.startswith(member_name) and (full_name[len(member_name)] in ['.', '[']):
                sub_struct = self[member]

                sub_name = full_name[len(member_name):]
                if sub_name.startswith('.'): sub_name = sub_name[1:]
                return sub_struct.get(sub_name)

            elif member_name==full_name:
                return self[member]

        raise KeyError

    def map(self, member, mapper, other, map_aux=False):
        if member not in self._members:
            return

        def map(x):
            y = x
            if nd.is_list(mapper):
                for m in mapper:
                    y = m(y)
            else:
                y = mapper(y)
            return y

        if self.is_data(member):
            other[member] = map(self[member])

        elif self.is_struct(member):
            sub_struct = self[member]
            other.make_struct(member)
            for sub_mem in list(sub_struct._members.keys()):
                sub_struct.map(sub_mem, mapper, other[member], map_aux)

        elif map_aux:
            other[member] = self._members[member]

    def copy(self, other, mapper=None):
        if mapper is None: mapper=lambda x: x

        for member in list(self._members.keys()):
            if self.is_data(member):
                other[member] = mapper(self[member])
            elif self.is_struct(member):
                other[member] = nd.Struct()
                self[member].copy(other[member], mapper)
            else:
                other[member] = self[member]

    def translate(self, old_name, new_name, mapper, remove=False):
        def map(x):
            y = x
            if nd.is_list(mapper):
                for m in mapper:
                    y = m(y)
            else:
                y = mapper(y)
            return y

        for member in list(self._members.keys()):
            if self.is_data(member):
                if member==old_name:
                    self[new_name] = map(self[old_name])
                    if remove: del self[old_name]

            elif self.is_struct(member):
                if member==old_name:
                    ns = nd.Struct()
                    self[old_name].copy(ns, mapper)
                    if remove: del self[old_name]
                    self[new_name] = ns
                else:
                    self[member].translate(old_name, new_name, mapper, remove)

    def translate_all(self, mapper):
        def map(x):
            y = x
            if nd.is_list(mapper):
                for m in mapper:
                    y = m(y)
            else:
                y = mapper(y)
            return y

        for member in list(self._members.keys()):
            if self.is_data(member):
                self[member] = map(self[member])

            elif self.is_struct(member):
                self[member].translate_all(mapper)

    def concat(self, other):
        dest = nd.Struct()
        for member in list(self._members.keys()):
            if self.is_data(member):
                dest[member] = nd.ops.concat(self[member], other[member])
            elif self.is_struct(member):
                dest[member] = self[member].concat(other[member])
            else:
                dest[member] = self[member]

        return dest
