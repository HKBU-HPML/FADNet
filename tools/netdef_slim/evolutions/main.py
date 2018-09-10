import netdef_slim as nd
from netdef_slim.core.register import register_function

nd.evo = None

_evolution_manager = nd.EvolutionManager()
nd.evo_manager = _evolution_manager

register_function('add_evo', _evolution_manager.add_evolution)
register_function('evo_names', _evolution_manager.evolution_names)
register_function('clear_evos', _evolution_manager.clear)

nd.evos = _evolution_manager._evolutions

def _select_evo(name):
    nd.evo = None
    for evo in _evolution_manager.evolutions():
        nd.log('check', evo.name(), name)
        if evo.name() == name:
            nd.log('<<< determined current evolution: %s >>>' % name)
            nd.evo = evo
    if nd.evo is None:
        raise BaseException('evolution %s not found' % name)

register_function('select_evo', _select_evo)

_training_dir = '.'


def _set_training_dir(dir):
    global _training_dir
    _training_dir = dir
    nd.evo_manager.set_training_dir(_training_dir)

register_function('set_training_dir', _set_training_dir)






