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
import os, socket
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
import datetime
import resource
from .trainerbase import TrainerBase
from .helpers import *
STOP_TIME = get_stop_time()

class SimpleTrainer(TrainerBase):
    """
    Simple trainer providing checkpoint management and a mainloop
    """

    def __init__(self, session, train_dir, signal_handling=True):
        """
        session: tf.Session
            The tensorflow session for training

        train_dir: str
            Directory used for storing logs and checkpoints during training.
        """
        super().__init__(session, train_dir, signal_handling)
        self._timeline_file = os.path.join(self._train_logdir, 'timeline.ctf.json')


    def _create_saver(self, max_to_keep, saver_var_list, checkpoints_path):
        """Returns the tf.train.Saver object

        max_to_keep: int
            Maximum number of snaphots to keep.

        saver_var_list: list or dict
            A list of variables to save or a dictionary which maps names to variables.
            The list or dict must contain the global_step

        Raises ValueError if the global_step is missing
        """
        assert isinstance(saver_var_list, (list,tuple,dict)), "saver_var_list must be a list or a dict"

        # check if the global step is in saver_var_list
        if isinstance(saver_var_list, (list,tuple)):
            if not self._global_step in saver_var_list:
                raise ValueError("global_step is not in saver_var_list")
        else: # dict
            if not self._global_step in saver_var_list.values():
                raise ValueError("global_step is not in saver_var_list")

        saver = tf.train.Saver(
            var_list=saver_var_list,
            max_to_keep=max_to_keep,
            )
        saver.recover_last_checkpoints([checkpoints_path])
        return saver




    def load_checkpoint(self, checkpoint_filepath=None, verbose=True, ignore_vars=None):
        """Restores variables from a checkpoint file.

        checkpoint_filepath: str
            The path to the checkpoint file.
            If None then the last saved checkpoint will be loaded.

        verbose: bool
            If True prints which variables will be restored or skipped

        """
        if checkpoint_filepath:
            print('loading', checkpoint_filepath, flush=True)
            optimistic_restore(self._session, checkpoint_filepath, verbose=verbose, ignore_vars=ignore_vars)
        else:
            checkpoints = retrieve_all_checkpoints(self._checkpoints_path) + retrieve_all_checkpoints(self._recovery_checkpoints_path)

            if checkpoints:
                last_checkpoint = sorted(checkpoints)[-1][1]
                print('loading', last_checkpoint, flush=True)
                optimistic_restore(self._session, last_checkpoint, verbose=verbose, ignore_vars=ignore_vars)
            else:
                print('nothing to restore. no checkpoint found.', flush=True)




    def mainloop(self,
            max_iter,
            train_ops,
            saver_interval,
            saver_max_to_keep=100000,
            saver_var_list=None,
            recovery_saver_interval=60,
            summary_int_ops=None,
            display_interval=100,
            display_str_ops=None,
            test_int_fn=None,
            custom_int_ops=None,
            runstats_interval=1000,
            trace_interval=33333,
            stop_time=STOP_TIME,
            ):
        """Standard main loop

        max_iter: int
            Maximum iteraion number.

        train_ops: list of ops
            List of training ops.

        saver_interval: int
            Number of iterations between checkpoints.

        saver_max_to_keep: int
            Maximum number of snaphots to keep.

        saver_var_list: list or dict
            A list of variables to save or a dictionary which maps names to variables.
            This parameter is directly passed to the tf.train.Saver
            The list or dict must contain the global_step.
            If None a default list with the global_step and all trainable variables
            will be created.

        recovery_saver_interval: float
            Time in minutes after last checkpoint which triggers saving a recovery checkpoint.

        summary_int_ops: list of tuple
            List of interval and operation tuples.
            E.g. [(100, summary1_op), (200, summary2_op)]

        display_interval: int
            Interval for running running display operations specified in 'display_str_ops'.

        display_str_ops: list of tuple
            List of string and operation tuples.
            E.g. [('MyLoss', op1), ('Ratio', op2)]

        test_int_fn: list of tuple
            List of interval and callable objects.
            E.g. [(1000, my_test_fn1), (1000, my_test_fn2)]
            The functions will be called before running the training ops

        custom_int_ops: list of tuple
            List of interval and operation.
            E.g. [(1000, my_op1), (1000, my_op2)]
            The ops will be run before running the training ops

        runstats_interval: int
            Interval for logging cpu/mem usage and iterations per second

        trace_interval: int
            Interval for writing a trace snapshot.


        stop_time:
            stop time in seconds since epoch.


        Returns a status code indicating if training was finished, crashed etc.
        """
        status_code = self.STATUS_TRAINING_UNFINISHED

        if saver_var_list is None:
            saver_var_list = create_save_var_dict()

        if summary_int_ops is None:
            summary_int_ops_str = []
        else:
            for idx, int_ops in summary_int_ops:
                summary_int_ops_str = [ (v[0],v[1],'_summary_{0}'.format(i)) for i,v in enumerate(summary_int_ops) ]

        if display_str_ops is None:
            display_str_ops = []

        if test_int_fn is None:
            test_int_fn = []

        if custom_int_ops is None:
            custom_int_ops = []

        # init savers and summary writer
        saver = self._create_saver(saver_max_to_keep, saver_var_list, self._checkpoints_path)
        recovery_saver = self._create_saver(2, saver_var_list, self._recovery_checkpoints_path)
        self._summary_writer = tf.summary.FileWriter(self._train_logdir,graph=tf.get_default_graph())


        # run ops for summaries and display more often in the beginning by using a power of 2 series
        # 1 2 4 8 .. interval
        def is_power_of_2(x):
            return x > 0 and ((x & (x - 1)) == 0)
        def run_op_in_current_iteration(iteration, interval):
            return (iteration % interval == 0) or (iteration < interval and is_power_of_2(iteration))


        start_iteration = self._session.run(self._global_step)
        global_step_value = start_iteration

        # mark the sessions start in the log
        self._summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START),global_step=start_iteration)


        last_snapshot_time = time.time()
        eta_time_start = time.time()
        disp_iter_timer = IterationTimer()
        summary_iter_timer = IterationTimer()
        netmon_iter_timer = IterationTimer()
        cpu_load_meter = CPULoadMeter()
        gpu_accounting = GPUAccounting()

        # always run these ops
        ops_always = train_ops
        train_ops_strings = ['_train_op{0}'.format(x) for x in range(len(train_ops))]
        ops_always_strings = train_ops_strings

        print( 'start iteration', start_iteration, flush=True)

        with self._coordinator.stop_on_exception():
            while global_step_value < max_iter and not self._coordinator.should_stop():
                if stop_time and time.time() > stop_time:
                    print("walltime reached",flush=True)
                    break

                # run test functions
                for interval, fn in test_int_fn:
                    if global_step_value % interval == 0:
                        fn()

                # run custom ops
                for interval, custom_op in custom_int_ops:
                    if global_step_value % interval == 0:
                        self._session.run(custom_op)

                # list of ops and op names for the current iteration
                ops = []
                op_strings = []

                # add ops that need to be run always
                ops.extend(ops_always)
                op_strings.extend(ops_always_strings)

                # append ops for summary and the display string
                for interval, summary_op, summary_name in summary_int_ops_str:
                    if run_op_in_current_iteration(global_step_value, interval):
                        ops.append(summary_op)
                        op_strings.append(summary_name)

                if run_op_in_current_iteration(global_step_value,display_interval):
                    for disp_name, disp_op in display_str_ops:
                        op_strings.append(disp_name)
                        ops.append(disp_op)

                # collect run meta data
                if global_step_value % trace_interval == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    #run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None


                # run all ops
                values = self._session.run(ops, options=run_options, run_metadata=run_metadata)
                
                # put values in a dict and get the new global step
                values_dict = dict(zip(op_strings,values))
                new_global_step_value = global_step_value+1


                # write run meta data
                if global_step_value % trace_interval == 0:
                    self._summary_writer.add_run_metadata(run_metadata, 'trace_step_{0}'.format(global_step_value))
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    with open(self._timeline_file, 'w') as trace_file:
                        trace_file.write(trace.generate_chrome_trace_format())



                # generate and print display string
                if global_step_value % display_interval == 0:
                    now = time.time()
                    time_per_iteration = disp_iter_timer.get_avg_iteration_time(int(global_step_value))
                    if not time_per_iteration:
                        print("# {0} {1:>8} | ".format(datetime.datetime.fromtimestamp(int(now)),global_step_value),end="")
                    else:
                        eta_iterations_per_second=(global_step_value-start_iteration)/(now-eta_time_start)
                        remaining_secs = int((max_iter-global_step_value)/eta_iterations_per_second)
                        remaining_time_str = str(datetime.timedelta(seconds=remaining_secs))

                        iterations_per_second = 1/time_per_iteration
                        print("# {0} {1:>8} {2:9.2f} ips  {3:>18} rem | ".format(
                                datetime.datetime.fromtimestamp(int(now)),
                                global_step_value,
                                iterations_per_second,
                                remaining_time_str),
                            end="")
                    for key, _ in display_str_ops:
                        value = values_dict[key]
                        if isinstance(value, np.float32):
                            print("{0}:{1:11.4g}  ".format(key,value).ljust(20),end="")
                        else:
                            print("{0}: {1}  ".format(key,value).ljust(20),end="")
                    print("",flush=True)


                # write summaries
                for interval, summary_op, summary_name in summary_int_ops_str:
                    if run_op_in_current_iteration(global_step_value, interval):
                        self._summary_writer.add_summary(values_dict[summary_name], global_step=global_step_value)

                if global_step_value % runstats_interval == 0:

                    # log some resource usage statistics and the current cpu load
                    rusage = resource.getrusage(resource.RUSAGE_SELF)
                    add_summary_simple_value(
                            self._summary_writer,
                            'runstats/maxrssMB',
                            global_step_value,
                            rusage.ru_maxrss//2**10)

                    add_summary_simple_value(
                            self._summary_writer,
                            'runstats/swaps',
                            global_step_value,
                            rusage.ru_nswap)

                    add_summary_simple_value(
                            self._summary_writer,
                            'runstats/fileInputs',
                            global_step_value,
                            rusage.ru_inblock)

                    add_summary_simple_value(
                            self._summary_writer,
                            'runstats/fileOutputs',
                            global_step_value,
                            rusage.ru_oublock)

                    add_summary_simple_value(
                            self._summary_writer,
                            'runstats/pageFaults_minor',
                            global_step_value,
                            rusage.ru_minflt)

                    add_summary_simple_value(
                            self._summary_writer,
                            'runstats/pageFaults_major',
                            global_step_value,
                            rusage.ru_majflt)

                    add_summary_simple_value(
                            self._summary_writer,
                            'runstats/contextSwitches_voluntary',
                            global_step_value,
                            rusage.ru_nvcsw)

                    add_summary_simple_value(
                            self._summary_writer,
                            'runstats/contextSwitches_involuntary',
                            global_step_value,
                            rusage.ru_nivcsw)

                    cpu_times = cpu_load_meter.get_avg_cpu_load()
                    if cpu_times:
                        avg_cpu_load = sum(cpu_times)
                        add_summary_simple_value(
                                self._summary_writer,
                                'runstats/cpuLoad',
                                global_step_value,
                                avg_cpu_load)
                        add_summary_simple_value(
                                self._summary_writer,
                                'runstats/cpuLoad_user',
                                global_step_value,
                                cpu_times[0])
                        add_summary_simple_value(
                                self._summary_writer,
                                'runstats/cpuLoad_sys',
                                global_step_value,
                                cpu_times[1])

                    # gpu stats
                    gpu_stats = gpu_accounting.get_accounting_stats()
                    if gpu_stats:
                        for gpu_idx, stat in gpu_stats.items():
                            keys = ('gpuUtilization', )#'memoryUtilization', 'maxMemoryUsage')
                            for k in keys:
                                add_summary_simple_value(
                                    self._summary_writer,
                                    'runstats/gpu{0}_{1}'.format(gpu_idx,k),
                                    global_step_value,
                                    stat[k] )


                    # log iterations per second
                    time_per_iteration = summary_iter_timer.get_avg_iteration_time(int(global_step_value))
                    if time_per_iteration:
                        add_summary_simple_value(
                                self._summary_writer,
                                'runstats/iterPerSec',
                                global_step_value,
                                float(1/time_per_iteration))

                # save checkpoints
                if global_step_value > start_iteration:
                    now = time.time()
                    if global_step_value % saver_interval == 0:
                        print("# {0}  saving.. ".format(datetime.datetime.fromtimestamp(int(time.time()))),end="")
                        checkpoint = saver.save(self._session, self._checkpoints_path, global_step=global_step_value)
                        print(checkpoint, flush=True)
                        last_snapshot_time = now

                    elif now-last_snapshot_time > 60*recovery_saver_interval:
                        print("# {0}  saving.. ".format(datetime.datetime.fromtimestamp(int(time.time()))),end="")
                        checkpoint = recovery_saver.save(self._session, self._recovery_checkpoints_path, global_step=global_step_value)
                        print(checkpoint, flush=True)
                        last_snapshot_time = now

                # send netmon package
                try:
                    time_per_iteration = netmon_iter_timer.get_avg_iteration_time(int(global_step_value))
                    ips = 1/time_per_iteration if time_per_iteration is not None else float('NaN')
                    m = ProcessStateMessage('TYPE=TENSORFLOW\n' +
                                            'HOST=%s\n' % socket.gethostname() +
                                            'USER=%s\n' % os.getenv('USER') +
                                            'PID=%d\n' % os.getpid() +
                                            'PBSID=%s\n' % os.getenv('PBS_JOBID', '') +
                                            'FOLDER=%s\n' % self._train_dir +
                                            'ITER=%d\n' % int(global_step_value) +
                                            'MAXITER=%d\n' % max_iter +
                                            'IPS=%.2f\n' % ips)
                    m.send_to('ororea', 10000)
                except Exception as e:
                    print(e)

                #if FLAGS.dry_run:
                    #break

                global_step_value = new_global_step_value

        print( 'stop iteration', global_step_value, flush=True )
        if global_step_value > start_iteration:
            print("# {0}  saving.. ".format(datetime.datetime.fromtimestamp(int(time.time()))),end="")
            if global_step_value == max_iter:
                checkpoint = saver.save(self._session, self._checkpoints_path, global_step=self._global_step)
            else:
                checkpoint = recovery_saver.save(self._session, self._recovery_checkpoints_path, global_step=self._global_step)
            print(checkpoint, flush=True)

        if global_step_value >= max_iter:
            status_code = self.STATUS_TRAINING_FINISHED


        return status_code

