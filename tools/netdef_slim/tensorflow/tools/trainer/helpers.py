#
#  tfutils - A set of tools for training networks with tensorflow
#  Copyright (C) 2017  Benjamin Ummenhofer
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import tensorflow as tf
import numpy as np
import os
import re
import time
import resource


def get_stop_time(time_buffer=5*60):
    """Retrieves the stop time from the environment variable STOP_TIME.

    time_buffer: int
        Time buffer in seconds. Default is 5 min to have enough time for the
        shutdown.

    Returns None if the variable has not been set.
    """
    if 'STOP_TIME' in os.environ:
        return int(os.environ['STOP_TIME'])-time_buffer
    else:
        return None



def add_summary_simple_value(writer, tag, step, simple_value):
    """Adds an event to the writer

    writer: tf.summary.FileWriter

    tag: str
        tag for the value

    step: int
        the global step

    simple_value: float or int
        A simple scalar float value
    """
    s = tf.Summary()
    s.value.extend([tf.Summary.Value(tag=tag,simple_value=simple_value)])
    writer.add_summary(s, global_step=step)


def read_global_step_from_checkpoint(save_file):
    """This function returns the global step for the checkpoint file.
    Returns None if there is no global step stored.

    save_file: str
        Path to the checkpoint without the .index, .meta or .data extensions.
    """
    reader = tf.train.NewCheckpointReader(save_file)
    if reader.has_tensor('global_step'):
        return reader.get_tensor('global_step')
    else:
        return None


def retrieve_all_checkpoints(checkpoint_path_prefix):
    """Retrieves a list of checkpoints for the prefix.

    checkpoint_path_prefix: str
        Prefix path without iteration number or file extension.

    Returns a sorted list of (iteration, checkpoint) tuples.
    E.g. '/bla/snapshot' could return [(100,'/tmp/bla-100'), (200,'/tmp/bla-200')].
    """
    checkpoints = []
    checkpoint_dir, file_prefix = os.path.split(checkpoint_path_prefix)
    if not os.path.isdir(checkpoint_dir):
        return []

    checkpoint_files = [ x for x in os.listdir(checkpoint_dir) if x.startswith(file_prefix) ]
    # print(checkpoint_files)
    iter_re = re.compile('.*-(\d+)\.data-(\d{5})-of-(\d{5})')
    for x in checkpoint_files:
        match = iter_re.match(x)
        if match:
            iteration = int(match.group(1))
            path = os.path.join(checkpoint_dir,file_prefix+'-'+match.group(1))

            # check if checkpoint is complete
            extensions = ['.index', '.meta']
            num_data = int(match.group(3))
            extensions += ['.data-{0:0>5d}-of-{1:0>5d}'.format(x,num_data) for x in range(num_data)]
            ok = True
            for ext in extensions:
                filepath = path+ext
                if not os.path.isfile(filepath) or os.stat(filepath).st_size == 0:
                    ok = False
                    break
            if ok:
                checkpoints.append((iteration,path))

    return sorted(checkpoints)


def list_evolution_checkpoints(train_dir, evolutions):
    """Returns a dict of (iteration, checkpoint) lists

    train_dir: str
        The path to the training dir
    evolutions: list of str
        List of evolution names

    Returns a dict which stores a sorted list of (iteration, checkpoint) tuples for
    each evolution.
    """
    from .trainerbase import TrainerBase
    result = {}
    for evo in evolutions:
        evo_checkpoint_path = os.path.join(
                train_dir,
                evo,
                TrainerBase.CHECKPOINTS_DIR,
                TrainerBase.CHECKPOINTS_FILE_PREFIX )

        result[evo] = retrieve_all_checkpoints(evo_checkpoint_path)

    return result



# function based on https://github.com/tensorflow/tensorflow/issues/312
def optimistic_restore_getvar(session, save_file, ignore_vars=None, verbose=False, ignore_incompatible_shapes=False):
    """This function tries to restore all variables in the save file.

    This function ignores variables that do not exist or have incompatible shape.
    Raises TypeError if the there is a type mismatch for compatible shapes.

    session: tf.Session
        The tf session

    save_file: str
        Path to the checkpoint without the .index, .meta or .data extensions.

    ignore_vars: list, tuple or set of str
        These variables will be ignored.

    verbose: bool
        If True prints which variables will be restored

    ignore_incompatible_shapes: bool
        If True ignores variables with incompatible shapes.
        If False raises a runtime error f shapes are incompatible.

    """
    def vprint(*args, **kwargs):
        if verbose: print(*args, flush=True, **kwargs)
    # def dbg(*args, **kwargs): print(*args, flush=True, **kwargs)
    def dbg(*args, **kwargs): pass
    if ignore_vars is None:
        ignore_vars = []

    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.dtype, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes and not var.name.split(':')[0] in ignore_vars])
    restore_vars = []

    dbg(saved_shapes)
    for v in tf.global_variables():
        dbg(v)

    nonfinite_values = False

    with tf.variable_scope('', reuse=True):
        for var_name, var_dtype, saved_var_name in var_names:
            dbg( var_name, var_dtype, saved_var_name, end='')
            curr_var = tf.get_variable(saved_var_name, dtype=var_dtype)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                dbg( ' shape OK')
                tmp = reader.get_tensor(saved_var_name)
                dbg(tmp.dtype)

                # check if there are nonfinite values in the tensor
                if not np.all(np.isfinite(tmp)):
                    nonfinite_values = True
                    print('{0} contains nonfinite values!'.format(saved_var_name), flush=True)

                if isinstance(tmp, np.ndarray):
                    saved_dtype = tf.as_dtype(tmp.dtype)
                else:
                    saved_dtype = tf.as_dtype(type(tmp))
                dbg(saved_dtype, var_dtype, saved_dtype.is_compatible_with(var_dtype))
                if not saved_dtype.is_compatible_with(var_dtype):
                    raise TypeError('types are not compatible for {0}: saved type {1}, variable type {2}.'.format(
                        saved_var_name, saved_dtype.name, var_dtype.name))

                vprint('restoring    ', saved_var_name)
                restore_vars.append(curr_var)
            else:
                vprint('not restoring',saved_var_name, 'incompatible shape:', var_shape, 'vs', saved_shapes[saved_var_name])
                if not ignore_incompatible_shapes:
                    raise RuntimeError('failed to restore "{0}" because of incompatible shapes: var: {1} vs saved: {2} '.format(saved_var_name, var_shape, saved_shapes[saved_var_name]))

    # not_found_in_checkpoint = sorted([(var.name, var.dtype, var.name.split(':')[0]) for var in tf.global_variables()
    #         if var.name.split(':')[0] not in saved_shapes and not var.name.split(':')[0] in ignore_vars])
    # for var_name, var_dtype, saved_var_name in not_found_in_checkpoint:
    #     vprint('not found in checkpoint    ', saved_var_name)

    global_names_split = [var.name.split(':')[0] for var in tf.global_variables()]
    not_found_in_graph = sorted([saved for saved in saved_shapes if saved not in global_names_split and not saved in ignore_vars])
    for var_name, var_dtype, saved_var_name in not_found_in_graph:
        vprint('not found in graph    ', saved_var_name)


    if nonfinite_values:
        raise RuntimeError('"{0}" contains nonfinite values!'.format(save_file))

    dbg( '-1-')
    saver = tf.train.Saver(
            var_list=restore_vars,
            restore_sequentially=True,)
    dbg( '-2-')
    saver.restore(session, save_file)
    dbg( '-3-')


# function based on https://github.com/tensorflow/tensorflow/issues/312
def optimistic_restore(session, save_file, ignore_vars=None, verbose=False, ignore_incompatible_shapes=False):
    """This function tries to restore all variables in the save file.

    This function ignores variables that do not exist or have incompatible shape.
    Raises TypeError if the there is a type mismatch for compatible shapes.

    session: tf.Session
        The tf session

    save_file: str
        Path to the checkpoint without the .index, .meta or .data extensions.

    ignore_vars: list, tuple or set of str
        These variables will be ignored.

    verbose: bool
        If True prints which variables will be restored

    ignore_incompatible_shapes: bool
        If True ignores variables with incompatible shapes.
        If False raises a runtime error f shapes are incompatible.

    """

    def vprint(*args, **kwargs):
        if verbose: print(*args, flush=True, ** kwargs)

    # def dbg(*args, **kwargs): print(*args, flush=True, **kwargs)
    def dbg(*args, **kwargs):
        pass

    if ignore_vars is None:
        ignore_vars = []

    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.dtype, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes and not var in ignore_vars])
    restore_vars = []

    dbg(saved_shapes)
    for v in tf.global_variables():
        dbg(v)

    nonfinite_values = False

    with tf.variable_scope('', reuse=True):
        for var_name, var_dtype, saved_var_name in var_names:
            dbg(var_name, var_dtype, saved_var_name, end='')
            #curr_var = tf.get_variable(saved_var_name, dtype=var_dtype)
            curr_var = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if saved_var_name in var.name][0]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                dbg(' shape OK')
                tmp = reader.get_tensor(saved_var_name)
                dbg(tmp.dtype)

                # check if there are nonfinite values in the tensor
                if not np.all(np.isfinite(tmp)):
                    nonfinite_values = True
                    print('{0} contains nonfinite values!'.format(saved_var_name), flush=True)

                if isinstance(tmp, np.ndarray):
                    saved_dtype = tf.as_dtype(tmp.dtype)
                else:
                    saved_dtype = tf.as_dtype(type(tmp))
                dbg(saved_dtype, var_dtype, saved_dtype.is_compatible_with(var_dtype))
                if not saved_dtype.is_compatible_with(var_dtype):
                    raise TypeError('types are not compatible for {0}: saved type {1}, variable type {2}.'.format(
                        saved_var_name, saved_dtype.name, var_dtype.name))

                vprint('restoring    ', saved_var_name)
                restore_vars.append(curr_var)
            else:
                vprint('not restoring', saved_var_name, 'incompatible shape:', var_shape, 'vs',
                       saved_shapes[saved_var_name])
                if not ignore_incompatible_shapes:
                    raise RuntimeError(
                        'failed to restore "{0}" because of incompatible shapes: var: {1} vs saved: {2} '.format(
                            saved_var_name, var_shape, saved_shapes[saved_var_name]))

    # not_found_in_checkpoint = sorted([(var.name, var.dtype, var.name.split(':')[0]) for var in tf.global_variables()
    #         if var.name.split(':')[0] not in saved_shapes and not var.name.split(':')[0] in ignore_vars])
    # for var_name, var_dtype, saved_var_name in not_found_in_checkpoint:
    #     vprint('not found in checkpoint    ', saved_var_name)

    # global_names_split = [var.name.split(':')[0] for var in tf.global_variables()]
    # not_found_in_graph = sorted(
    #     [saved for saved in saved_shapes if saved not in global_names_split and not saved in ignore_vars])
    # for var_name, var_dtype, saved_var_name in not_found_in_graph:
    #     vprint('not found in graph    ', saved_var_name)

    if nonfinite_values:
        raise RuntimeError('"{0}" contains nonfinite values!'.format(save_file))

    dbg('-1-')
    saver = tf.train.Saver(
        var_list=restore_vars,
        restore_sequentially=True, )
    dbg('-2-')
    saver.restore(session, save_file)
    dbg('-3-')


def create_save_var_dict( collections=(tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.TRAINABLE_VARIABLES) ):
    """Creates a dictionary that maps names to saveable objects.
    This dict can be directly passed to a tf.train.Saver

    collections: list of str
        List of collections which will be added to the dict.
        Default is the global_step and all trainable variables.

    Returns a dict that maps names to saveable objects
    """
    save_vars = []
    for c in collections:
        save_vars.extend(tf.get_collection(c))
    return {v.op.name: v for v in save_vars}



def get_gpu_count():
    """Returns the number of visible gpus using nvml.
    Returns None if nvml is not available.
    """
    from .nvml import HAVE_NVML, nvmlDeviceGetCount
    if HAVE_NVML:
        return nvmlDeviceGetCount()
    return None



# based on the function from the cifar10 tutorial
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  if len(tower_grads) == 1: # no averaging if there is only one item
    return tower_grads[0]

  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads



def combine_loss_dicts(list_of_loss_dicts, average=True):
    """Sums up or averages losses with the same name across all dictionaries

    list_of_loss_dicts: list of dict of (name, tensor)
        A list of dicionaries with scalar losses

    average: bool
        If True computes the average instead of the sum.

    Returns a dictionary that sums up the respective losses
    """
    if len(list_of_loss_dicts) == 1:
        return list_of_loss_dicts[0]

    loss_names = set()
    for ld in list_of_loss_dicts:
        for k in ld:
            loss_names.add(k)

    result = {}
    for name in loss_names:
        losses = []
        for ld in list_of_loss_dicts:
            if name in ld:
                losses.append(ld[name])
        if len(losses) > 1:
            if average:
                result[name] = tf.add_n(losses, name)/len(losses)
            else:
                result[name] = tf.add_n(losses, name)
        else:
            result[name] = losses[0]

    return result



class IterationTimer:
    def __init__(self):
        self.start_iteration = None
        self.start_time = None

    def get_avg_iteration_time(self, iteration):
        """Returns the averaged time per iteration since the last call

        iteration: int
            The current iteration

        Returns the average time per iteration or None
        """
        if not self.start_iteration or self.start_iteration >= iteration:
            self.start_iteration = iteration
            self.start_time = time.time()
            return None
        else:
            now = time.time()
            avg_iteration_time = (now - self.start_time)/(iteration-self.start_iteration)
            self.start_iteration = iteration
            self.start_time = now
            return avg_iteration_time


class CPULoadMeter:
    def __init__(self):
        self.start_cpu_user_time = None
        self.start_cpu_sys_time = None
        self.start_wall_time = None

    def get_avg_cpu_load(self):
        """Returns the average cpu load since the last call

        Returns the user and system time fraction per second as tuple or None
        """
        if not self.start_wall_time:
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            self.start_wall_time = time.time()
            self.start_cpu_user_time = rusage.ru_utime
            self.start_cpu_sys_time = rusage.ru_stime
            return None
        else:
            now = time.time()
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            time_delta = now-self.start_wall_time
            avg_user_time = (rusage.ru_utime-self.start_cpu_user_time)/time_delta
            avg_sys_time = (rusage.ru_stime-self.start_cpu_sys_time)/time_delta
            self.start_wall_time = now
            self.start_cpu_user_time = rusage.ru_utime
            self.start_cpu_sys_time = rusage.ru_stime
            return avg_user_time, avg_sys_time



class GPUAccounting:
    _initialized = False
    device_handles = {}
    device_handles_with_accounting = {}
    pid = os.getpid()

    def __init__(self):
        from .nvml import HAVE_NVML, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetAccountingMode, NVML_FEATURE_ENABLED
        if not GPUAccounting._initialized and HAVE_NVML:
            dev_count = nvmlDeviceGetCount()
            for idx in range(dev_count):
                device = nvmlDeviceGetHandleByIndex(idx)
                GPUAccounting.device_handles[idx] = device
                if nvmlDeviceGetAccountingMode(device) == NVML_FEATURE_ENABLED:
                    GPUAccounting.device_handles_with_accounting[idx] = device
            GPUAccounting._initialized = True

    def get_accounting_stats(self):
        """Returns the accounting stats for all gpus for the pid
        Returns an empty dict if there is no gpu or accounting is disabled for all gpus.
        """
        from .nvml import nvmlDeviceGetAccountingStats
        result = {}
        for idx, device in GPUAccounting.device_handles_with_accounting.items():
            stats = nvmlDeviceGetAccountingStats(device, GPUAccounting.pid)
            if not stats is None:
                result[idx] = stats
        return result


