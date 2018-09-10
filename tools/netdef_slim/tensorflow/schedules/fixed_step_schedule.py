from netdef_slim.schedules.fixed_step_schedule import _FixedStepSchedule
from netdef_slim.core.register import register_class
import tensorflow as tf

nothing = None

class FixedStepSchedule(_FixedStepSchedule):
    def __init__(self, name, base_lr, max_iter, steps, gamma=0.5, stretch=1.0):
        super().__init__(name, base_lr, max_iter, steps, gamma, stretch)

    def get_schedule(self, global_step):
        lr = self.base_lr()
        lr_steps = [lr]
        for step in self.step_iters():
            lr = lr * self.gamma()
            lr_steps.append(lr)
        return  tf.train.piecewise_constant(global_step, self.step_iters(), lr_steps)

register_class('FixedStepSchedule', FixedStepSchedule)