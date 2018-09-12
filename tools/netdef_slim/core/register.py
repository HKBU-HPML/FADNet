import netdef_slim
import importlib

def _register(name, x):
    if '.' not in name:
        setattr(netdef_slim,  name, x)
    else:
        parts = ['netdef_slim'] + name.split('.')
        module = '.'.join(parts[:-1])
        mod = importlib.import_module(module)
        setattr(mod,  parts[-1], x)

def register_op(name, function):
    setattr(netdef_slim.ops,  name, function)

def register_class(name, cl):
    _register(name, cl)

def register_function(name, function):
    _register(name, function)

_chosen_framework = None

def chosen_framework():
    return _chosen_framework

def require_chosen_framework(module):
    if _chosen_framework is None: raise BaseException('framwork required for module %s' % module)
    return _chosen_framework

def choose_framework(framework):
    global _chosen_framework

    if _chosen_framework is not None and _chosen_framework!=framework:
        raise BaseException('attempt to switch already chosen framework')

    _chosen_framework = framework

    if framework == 'caffe':         import netdef_slim.caffe
    elif framework == 'tensorflow':  import netdef_slim.tensorflow
    else:                            raise NotImplementedError

