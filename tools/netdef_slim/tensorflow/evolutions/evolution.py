import netdef_slim as nd
import os, re
from netdef_slim.core.register import register_class
from netdef_slim.evolutions.evolution import _Evolution as BaseEvolution
from netdef_slim.tensorflow.evolutions.state import _State

nothing = None

class _Evolution(BaseEvolution):
    def __init__(self, training_dataset, validation_datasets, schedule, params={}, name=None):
        super().__init__(training_dataset, validation_datasets, schedule, params=params, name=name)

    def _snapshots_path(self):
        return os.path.join(self.path(), 'checkpoints')

    def has_folder(self):
        return os.path.exists(self._snapshots_path())

    def make_folder(self):
        if not os.path.exists(self.path()):
            os.mkdir(self.path(), mode=0o755)
            os.mkdir(self._snapshots_path(), mode=0o755)

    def states(self):
        found_states = []
        data_file_re = re.compile('([^0-9]*(\d+))\.data-(\d{5})-of-(\d{5})')

        for filename in os.listdir(self._snapshots_path()):
            match = data_file_re.match(filename)
            if match:
                # print('#### ' + filename + ' MATCH')
                # check if checkpoint has all files
                extensions = ['.index', '.meta']
                num_data = int(match.group(3))
                extensions += ['.data-{0:0>5d}-of-{1:0>5d}'.format(x, num_data) for x in range(num_data)]
                ok = True
                for ext in extensions:
                    f = os.path.join(self._snapshots_path(), match.group(1)+ext)
                    if not os.path.isfile(f) or os.stat(f).st_size == 0:
                        ok = False
                        break

                if ok:
                    #print('### DBG match group 2:  ' + match.group(2)+',   '+filename)
                    found_states.append(_State(self.name()+':'+match.group(2)))
            # else: print('#### '+filename+' NO MATCH')

        return sorted(found_states, key=lambda state: state.iter())

    def prefix(self):
        file_re = re.compile('([^0-9]*)[0-9]+\.(index|meta|data-\d{5}-of-\d{5})')
        for file in os.listdir(self._snapshots_path()):
            match = file_re.match(file)
            if match:
                return match.group(1)

        return ''

    def check_states_log(self):
        states = self.states()
        if len(states):
            try:
                checkpoint_log = open(os.path.join(self._snapshots_path(), 'checkpoint'), 'r+')
            except FileNotFoundError:
                return nd.status.DATA_MISSING
            last_state = states[-1]
            log_line_re = re.compile('model_checkpoint_path: "'+self.prefix()+str(last_state.iter())+'"')

            line=checkpoint_log.readline()
            match = log_line_re.match(line)
            if not match:
                return nd.status.CONFIGURATION_ERROR
        if not len(states):
            if os.path.isfile(os.path.join(self._snapshots_path(), 'checkpoint')) and os.path.getsize(os.path.join(self._snapshots_path(), 'checkpoint'))>0:
                return nd.status.CONFIGURATION_ERROR

        return nd.status.SUCCESS

    def update_states_log(self):
        checkpoint_log = open(os.path.join(self._snapshots_path(), 'checkpoint'), 'w')
        last_state = self.last_state()
        if last_state is not None:
            checkpoint_log.write('model_checkpoint_path: \"' + self.prefix() + str(last_state.iter()) + '\"\n')

    def get_state_path(self, id):
        return os.path.join(self._snapshots_path(), self.prefix()+str(id))

register_class('Evolution', _Evolution)
