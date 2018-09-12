import sys, os

def is_list(x):
    return isinstance(x, (list, tuple)) and not isinstance(x, str)


def make_list(x):
    if not is_list(x):
        return (x,)
    return x

_quiet = False

if 'NETDEF_QUIET' in os.environ and os.environ['NETDEF_QUIET'] == 'True':
    _quiet = True

def set_quiet(value):
    global _quiet
    _quiet = value

def log_message(*args):
    if _quiet: return

    s = ''
    for arg in args:
        if s != '': s += ' '
        s += str(arg)

    sys.stderr.write(s)
    sys.stderr.write('\n')
    sys.stderr.flush()

log = log_message

def merge(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def load_module(path):
    import sys, os
    from importlib.machinery import SourceFileLoader

    oldpath = sys.path
    sys.path.append(os.path.dirname(path))
    mod = SourceFileLoader("None", path).load_module()
    sys.path.remove(os.path.dirname(path))

    return mod

