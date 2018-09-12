import netdef_slim as nd
import os, re

class _Evolution:
    def __init__(self, training_dataset, validation_datasets, schedule, params={}, name=None):
        self._training_dataset = training_dataset
        self._validation_datasets = validation_datasets
        if isinstance(schedule, str):
            schedule = nd.get_default_schedule(schedule)
        self._schedule = schedule
        self._params = params # key value parameter to specify differently for each evo
        self._name = name

    def set_name(self, name): self._name = name

    def get_param_value(self, key):
        if key in self._params.keys():
            return self._params[key]
        else:
            raise KeyError('%s is not found' % key)


    def set_index(self, index): self._index = index

    def training_dataset(self): return self._training_dataset

    def validation_datasets(self): return self._validation_datasets

    def schedule(self): return self._schedule

    def max_iter(self): return self._schedule.max_iter()

    def name(self): return self._name

    def index(self): return self._index

    def is_complete(self):
        last_state = self.last_state()
        if last_state is not None:
            return self.max_iter() == last_state.iter()
        return False

    def has_folder(self):
        return os.path.exists(self.path())

    def make_folder(self):
        if not os.path.exists(self.path()):
            os.mkdir(self.path(), mode=0o755)

    def __lt__(self, other):
        return (self.index() < other.index())

    def __le__(self, other):
        return (self.index() <= other.index())

    def __gt__(self, other):
        return (self.index() > other.index())

    def __ge__(self, other):
        return (self.index() >= other.index())

    def __eq__(self, other):
        return (self.index() == other.index())

    def __ne__(self, other):
        return (self.index() != other.index())

    def path(self):
        return os.path.join(nd.evo_manager.training_dir(), self.name())

    def last_state(self):
        states = self.states()
        if len(states):
            return states[-1]

    def has_states(self):
        return len(self.states()) > 0

    def clean(self):
        for state in self.states():
            state.clean()
        self.update_states_log()

    def clean_after(self, state):
        for s in self.states():
            if s > state: s.clean()
        self.update_states_log()

    def last_snapshot_path(self):
        last_state = self.last_state()
        if last_state is not None:
            return last_state.path()

    def last_snapshot_iter(self):
        last_state = self.last_state()
        if last_state is not None:
            return last_state.iter()

    def get_state(self, iter):
        for state in self.states():
            if state.iter() == int(iter):
                return state

    def __str__(self):
        return '%s        %s' % (self.name(), 'COMPLETE' if self.is_complete() else '')
