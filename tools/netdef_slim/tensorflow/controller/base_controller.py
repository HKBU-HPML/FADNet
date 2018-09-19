import netdef_slim as nd
nd.choose_framework('tensorflow')

import argparse, re, datetime, sys, os
import tensorflow as tf
from netdef_slim.tensorflow.controller.net_actions import NetActions
from netdef_slim.utils import io
import signal



class BaseTFController:
    base_path=None

    def __init__(self, path=None, net_actions=NetActions):
        if path is not None:
            self.base_path = path
        nd.load_module(os.path.join(self.base_path, 'config.py'))
        self.net_actions = net_actions

    def run(self):
        self._command_hooks = {}
        self._parser = argparse.ArgumentParser(description='process network')
        self._configure_parser()
        self._configure_subparsers()
        self._args = self._parser.parse_args()

        command = self._args.command
        if command is None:
            self._parser.print_help()
            return

        if command not in self._command_hooks:
            raise BaseException('Unknown command: ' + command)

        self._command_hooks[command]()

    def _configure_parser(self):
        self._parser.add_argument('--gpu-id',    help='outside cluster: gpu ID to use (default=0)', default=None, type=int)
        self._subparsers = self._parser.add_subparsers(dest='command', prog='controller')

    def _configure_subparsers(self):
        # eval
        subparser = self._subparsers.add_parser('eval', help='run network on images')
        subparser.add_argument('img0',        help='path to input img0')
        subparser.add_argument('img1',        help='path to input img1')
        subparser.add_argument('out_dir',        help='path to output dir')
        subparser.add_argument('--state',     help='state of the snapshot', default=None)
        def eval():
            self.eval(image_0=self._args.img0,
                      image_1=self._args.img1,
                      out_dir=self._args.out_dir,
                      state=self._args.state)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['eval'] = eval

        # perf_test
        subparser = self._subparsers.add_parser('perf-test', help='measure of runtime of core net with no data I/O')
        subparser.add_argument('--burn_in',     help='number of iters to burn-in before measureing runtime', default=50)
        subparser.add_argument('--iters',       help='number of iters to average runtime', default=100)
        subparser.add_argument('--resolution',  help='the resolution used to measure runtime (width height), default is the Sintel resolution', nargs=2, default=(1024, 436))
        def perf_test():
            self.perf_test(burn_in = self._args.burn_in,
                           iters=self._args.iters,
                           resolution=self._args.resolution)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['perf-test'] = perf_test


        # list-evos
        subparser = self._subparsers.add_parser('list-evos', help='list evolution definitions')
        def list_evos():
            for evo in nd.evo_manager.evolutions():
                print(evo)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['list-evos'] = list_evos

        # list-states
        subparser = self._subparsers.add_parser('list-states', help='list present states')
        def list_states():
            for evo in nd.evo_manager.evolutions():
                for state in evo.states():
                    print(state)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['list-states'] = list_states


    def eval(self, **kwargs):
        out_dir = kwargs.pop('out_dir')
        output = self.net_actions(net_dir=self.base_path).eval(**kwargs)
        print('Saving output in: {}'.format(out_dir))
        for k,v in output.items():
            suffix = '.float3'
            if 'flow' in k:
                suffix='.flo'
            out_path = os.path.join(out_dir, k+suffix)
            io.write(out_path, v[0,:,:,:].transpose(1,2,0))
        return output
