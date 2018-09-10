
from .schedule import _Schedule

class _FixedStepSchedule(_Schedule):
    def __init__(self, name, base_lr, max_iter, steps, gamma=0.5, stretch=1.0):
        super().__init__(name, base_lr, max_iter, stretch)
        self._gamma = gamma
        self._step_iters = steps

    def step_iters(self):
        steps = []
        for iter in self._step_iters:
            steps.append(int(self._stretch*iter))
        return steps

    def gamma(self): return self._gamma
