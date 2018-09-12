import netdef_slim as nd
from netdef_slim.core.register import register_class
from netdef_slim.evolutions.evolution_manager import _EvolutionManager as BaseEvolutionManager
from netdef_slim.evolutions.state import _State

nothing = None

class _EvolutionManager(BaseEvolutionManager):
    def __init__(self):
        super().__init__()

    def get_state(self, id):
        if isinstance(id, _State):
            return id
        elif id is None:
            return None
        elif ':' in str(id):
            evo_name, state_name = id.split(':')
            state = self.get_evolution(evo_name).get_state(state_name)
            if state is None:
                raise KeyError('State <'+str(id)+'> not found')
            return state
        else:
            try:
                id = int(id)
                for evo in self.evolutions():
                    state = evo.get_state(id)
                    if state is not None:
                        return state
            except ValueError:
                pass

            raise Exception('State '+str(id)+' not found.')
            # if len(self.evolutions())==1:
            #     return self.first_evolution().get_state(id)
            # else:
            #     raise ValueError('Cannot identify state '+str(id)+' with more than 1 evolution, specify one.')

register_class('EvolutionManager', _EvolutionManager)