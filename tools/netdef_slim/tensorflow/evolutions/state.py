import netdef_slim as nd
import re, os
from netdef_slim.core.register import register_class
from netdef_slim.evolutions.state import _State as BaseState

nothing = None

class _State(BaseState):
    def __init__(self, id):
        super().__init__(id)

    def files(self):
        index = None
        meta = None
        data = []

        file_re = re.compile('([^0-9]*)'+str(self.iter())+'{1}(.*)')
        for file in os.listdir(self.folder()):
            file_match = file_re.match(file)
            if file_match:
                if re.compile('\.index').match(file_match.group(2)):
                    index = os.path.join(self.folder(), file)
                elif re.compile('\.meta').match(file_match.group(2)):
                    meta = os.path.join(self.folder(), file)
                elif re.compile('\.data-\d{5}-of-\d{5}').match(file_match.group(2)):
                    data.append(os.path.join(self.folder(), file))
                prefix = file_match.group(1)
        if index is not None and meta is not None and len(data) > 0:
            #return data + [index] + [meta]
            return {'data': data, 'index': index, 'meta': meta, 'prefix': prefix}

    def folder(self):
        return os.path.join(nd.evo_manager.training_dir(), self.evo_name(), 'checkpoints')

    def clean(self):
        files = self.files()
        if files is not None:
            os.remove(files['index'])
            os.remove(files['meta'])
            [os.remove(data_file) for data_file in files['data']]

    def path(self):
        return os.path.join(self.folder(), self.files()['prefix']+str(self.iter()))

register_class('State', _State)