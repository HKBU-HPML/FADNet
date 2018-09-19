from copy import deepcopy
from netdef_slim.core.register import register_function
import netdef_slim as nd


class BaseScope(object):
    def __init__(self, name=None, **kwargs):
        global scope_stack, current_scope
        self._name = name
        self._parent = None
        self._config = {}

        if len(scope_stack):
            self._parent = nd.scope
        push_scope(self)

        self._config = deepcopy(kwargs)
        if self._parent:
            for key, value in self._parent.config().items():
                if key not in self._config:
                    self._config[key] = deepcopy(value)

    def __enter__(self):
        pass

    def __exit__(self, type, val, tb):
        pop_scope()

    def param(self): return self._config['param']
    def loss_fact(self): return self._config['loss_fact']

    def config(self): return self._config
    def parent(self): return self._parent
    def name(self): return self._name

    def full_name(self, member=None):
        name = self.name()
        if self._parent is not None:
            parent_name = self._parent.full_name()
            if parent_name is not None: name = parent_name + '/' + name

        if member is not None and name is not None:
            name = name + '/' + member

        if name is None:
            name = member

        return name

    def conv(self, *args, **kwargs): return self._config['conv_op'](*args, **kwargs)
    def conv_nl(self, *args, **kwargs): return self._config['conv_nonlin_op'](*args, **kwargs)
    def upconv(self, *args, **kwargs): return self._config['upconv_op'](*args, **kwargs)
    def upconv_nl(self, *args, **kwargs): return self._config['upconv_nonlin_op'](*args, **kwargs)


scope_stack = []

nd.scope = None

def push_scope(scope):
    global scope_stack, current_scope
    scope_stack.append(scope)
    nd.scope = scope

def pop_scope():
    global scope_stack, current_scope
    scope = scope_stack[-1]
    scope_stack = scope_stack[:-1]
    nd.scope = scope_stack[-1]
    return scope

bottom_scope = None

