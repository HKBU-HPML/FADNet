nothing = None

import netdef_slim as nd
from netdef_slim.core.register import register_function

def _get_default_schedule(name, stretch=1.0):
    sel_name = name
    if name.endswith('_half'):
        stretch *= 0.5
        sel_name = name.replace('_half', '')

    if sel_name == 'S_pretrain':
        return nd.FixedStepSchedule(
            name = name,
            base_lr = 0.0001,
            steps = [],
            max_iter=100000,
            stretch=stretch
        )
    elif sel_name == 'S_short':
        return nd.FixedStepSchedule(
            name = name,
            base_lr = 0.0001,
            steps = [300000, 400000, 500000],
            max_iter=600000,
            stretch=stretch
        )
    elif sel_name == 'S_long':
        return nd.FixedStepSchedule(
            name = name,
            base_lr = 0.0001,
            steps = [400000, 500000, 600000, 800000, 1000000],
            max_iter=1200000,
            stretch=stretch
        )
    elif sel_name == 'S_fine':
        return nd.FixedStepSchedule(
            name = name,
            base_lr = 0.00001,
            steps = [200000, 300000, 400000],
            max_iter=500000,
            stretch=stretch
        )
    elif sel_name == 'S_fine_sd':
        return nd.FixedStepSchedule(
            name = name,
            base_lr = 0.00001,
            steps = [150000, 200000, 250000],
            max_iter=300000,
            stretch=stretch
        )
    elif sel_name == 'S_refinement':
        return nd.FixedStepSchedule(
            name = name,
            base_lr = 0.0001,
            steps = [100000, 125000, 150000, 175000],
            max_iter=200000,
            stretch=stretch
        )
    elif sel_name == 'S_experimental':
        return nd.FixedStepSchedule(
            name = name,
            base_lr = 0.0001,
            steps = [100000, 150000, 200000, 250000],
            max_iter=300000,
            stretch=stretch
        )
    elif sel_name == 'S_experimental_init_phase':
        return nd.FixedStepSchedule(
            name = name,
            base_lr = 0.0001,
            steps = [100000],
            max_iter=100000,
            stretch=stretch
        )
    elif sel_name == 'S_experimental_second_phase':
        return nd.FixedStepSchedule(
            name = name,
            base_lr = 0.00005,
            steps = [50000, 100000, 150000, 200000],
            max_iter=200000,
            stretch=stretch
        )
    elif sel_name == 'S_none':
        return nd.FixedStepSchedule(
            name = name,
            base_lr = 0.0001,
            max_iter=100000000,
            stretch=stretch
        )

    raise Exception('Default schedule %s does not exist.' % name)


register_function('get_default_schedule', _get_default_schedule)
