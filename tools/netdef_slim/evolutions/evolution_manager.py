import netdef_slim as nd
from netdef_slim.evolutions.state import _State
import os
nothing = None

class _EvolutionManager:
    def __init__(self, training_dir='.'):
        self._evolutions = []
        self._training_dir = training_dir

    def clear(self):
        self._evolutions = []
        self._training_dir = '.'

    def get_status(self):
        # evo = self.current_evolution()
        return self.last_trained_evolution(), self.current_evolution()
        # return evo.path(), evo.training_dataset(), evo.validation_datasets(), evo.schedule(), self.last_trained_evolution().last_snapshot_path()

    def set_training_dir(self, training_dir):
        self._training_dir = training_dir
        self.make_training_dir()

    def make_training_dir(self):
        if not os.path.exists(self.training_dir()):
            os.mkdir(self.training_dir(), mode=0o755)

    def training_dir(self):
        return self._training_dir

    def add_evolution(self, evolution):
        idx = len(self.evolutions())
        evolution.set_index(idx)
        if evolution.name() is None:
            evolution.set_name('%02d__%s__%s' % (idx, evolution.training_dataset(), evolution.schedule().name()))
        evolution.make_folder()
        self._evolutions.append(evolution)

        ds_name = evolution.training_dataset()
        if isinstance(evolution.training_dataset(), dict):
            ds_name = ''
            for key in evolution.training_dataset():
                if ds_name != '': ds_name += '_'
                ds_name += key

        nd.log('evolution <%s>: %s' % (evolution.name(), str(evolution)))

    def evolutions(self):
        return self._evolutions

    def evolution_names(self):
        return [evo.name() for evo in self._evolutions]

    def get_evolution(self, id):
        if isinstance(id, int):
            return self._evolutions[id]
        elif isinstance(id, str):
            for evo in self._evolutions:
                if evo.name() == id:
                    return evo

    def evolution_name(self, index):
        return self.get_evolution(index).name()

    def evolution_index(self, name):
        return self.get_evolution(name).index()

    def first_evolution(self):
        return self.get_evolution(0)

    def last_evolution(self):
        return self.get_evolution(-1)

    def last_trained_evolution(self):
        for evo in reversed(self._evolutions):
            if evo.has_states():
                return evo
        return self.first_evolution()

    def current_evolution(self):
        last_trained = self.last_trained_evolution()
        if not last_trained.is_complete():
            print('Evo manager: '+last_trained.name()+' not complete')
            return last_trained
        else:
            if not self.is_complete():
                print('Evo manager: '+last_trained.name() + ' complete, getting new')
                return self.get_evolution(last_trained.index()+1)
            else:
                print('Evo manager: complete')
                return

    def is_complete(self):
        return self.last_trained_evolution().index() is self.last_evolution().index()

    def get_state(self, id):
        if isinstance(id, _State):
            return id
        elif ':' in str(id):
            evo_name, state_name = id.split(':')
            state = self.get_evolution(evo_name).get_state(state_name)
            if state is None:
                raise KeyError('State <' + str(id) + '> not found')
            return state
        else:
            if len(self.evolutions())==1:
                return self.first_evolution().get_state(id)
            else:
                raise ValueError('Cannot identify state '+str(id)+' with more than 1 evolution, specify one.')

    def get_last_present_state(self):
        for evo in reversed(self.evolutions()):
            last_evo_state = evo.last_state()
            if last_evo_state is not None:
                return last_evo_state

        return None

    def clean(self):
        for evo in self.evolutions():
            evo.clean()

    def clean_after(self, state):
        for evo in self.evolutions():
            evo.clean_after(self.get_state(state))

    def existing_data(self, state=None):
        if state is None:
            for evo in self.evolutions():
                if evo.has_states():
                    return True
        else:
            state = self.get_state(state)
            for evo in self.evolutions():
                for s in evo.states():
                    if s > state:
                        return True
        return False

    def check_train(self):
        for evo in self.evolutions():
            if not evo.has_folder(): return nd.status.DATA_MISSING
            if not evo.check_states_log(): return nd.status.DATA_MISSING

        return nd.status.SUCCESS
        # TODO what else?

    def make_folders(self):
        for evo in self.evolutions():
            evo.make_folder()
