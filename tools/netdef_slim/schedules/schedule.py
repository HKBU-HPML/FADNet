
class _Schedule:
    def __init__(self, name, base_lr, max_iter, stretch=1.0):
        self._name = name
        self._base_lr = base_lr
        self._max_iter = max_iter
        self._stretch = stretch

    def set_stretch(self, stretch): self._stretch = stretch
    def name(self): return self._name
    def base_lr(self): return self._base_lr
    def max_iter(self): return int(self._stretch*self._max_iter)
    def lr(self, iter):
        raise NotImplemented
