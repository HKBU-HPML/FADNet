import sys
import time
import uuid
import os
import argparse
import shutil
import netdef_slim as nd

class Controller:
    base_path = None

    def __init__(self, path=None, gpu_id=0, quiet=False):
        if path is None:
            path = self.base_path

        self._gpu_id = gpu_id
        self._path = path
        self._quiet = quiet

        parts = list(os.path.normpath(self._path).split('/'))
        self._name = parts[-1]

        self._train_dir = self.path('training')
        self._scratch_dir = 'scratch/%s' % uuid.uuid4()
        self._scratch_log_file = self._scratch_dir + '/log.txt'

        self._net_config_file = self.path('config.py')

        os.environ['NETDEF_QUIET'] = str(quiet)
        nd.set_quiet(quiet)
        nd.load_module(self._net_config_file)

