from netdef_slim.schedules.schedule import _Schedule

nothing = None

class Schedule(_Schedule):
    def __init__(self, name, base_lr, max_iter, stretch=1.0):
        super().__init__(name, base_lr, max_iter, stretch)