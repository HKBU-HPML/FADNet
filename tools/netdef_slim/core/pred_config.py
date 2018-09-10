import netdef_slim as nd


class PredConfigId:
    def __init__(self, type, channels, dir=None, offset=None, perspective=None, scale=1.0, dist=None, mod_func=None, array_length=0):
        self.type = type
        self.dir = dir
        self.offset = offset
        self.perspective = perspective
        self.dist = dist
        self.perspective = perspective
        self.channels = channels
        self.scale = scale
        self.mod_func = mod_func
        self.array = False if array_length==0 else True
        self.array_length = array_length if self.array else 1
        self.total_channels = self.channels * self.array_length

    def __str__(self):
        return 'PredConfigId(type=\'%s\', dir=\'%s\', offset=%s, perspective=%s, dist=%s, channels=%s, scale=%s, mod_func=%s, array_length=%d)' % (
            self.type,
            self.dir,
            self.offset,
            self.perspective,
            self.dist,
            self.channels,
            self.scale,
            self.mod_func,
            self.array_length
        )


class PredConfig:
    def __init__(self, ids=None, multiframe=False):
        self._ids = ids if ids is not None else []
        self._multiframe = multiframe

    def add(self, id):
        self._ids.append(id)

    def clear(self):
        self._ids.clear()

    def channel_counts(self):
        channel_counts = []

        for id in self._ids:
            if id.array:
                for i in range(0, id.array_length):
                    channel_counts.append(id.channels)
            else:
               channel_counts.append(id.channels)

        return channel_counts

    def __getitem__(self, item):
        return self._ids[item]

    def slice_config(self):
        slice_points = []
        current = 0
        for c in self.channel_counts()[:-1]:
            current += c
            slice_points.append(current)

        return slice_points

    def total_channels(self):
        num = 0
        for c in self.channel_counts():
            num += c
        return num

    def disassemble(self, prediction, **kwargs):
        config = self.slice_config()
        if len(config) == 0: slices = [prediction, ]
        else:                slices = nd.ops.slice(prediction, self.slice_config())

        i = 0
        def get_slice(scale):
            nonlocal i
            s = slices[i]
            i += 1
            if scale != 1: s = nd.ops.scale(s, scale)
            if id.mod_func is not None: s = id.mod_func(s, **kwargs)
            return s

        data = nd.Struct()
        for id in self._ids:
            if id.dist is not None and id.dist>1 and not self._multiframe:
                raise BaseException('found a dist > 1 in data, but multiframe is False')
            if self._multiframe:
                raise NotImplementedError

            s = data
            n = None
            p = None # parent

            def descend(name):
                nonlocal s, n, p
                if name is not None:
                    s.make_struct(name)
                    p = s
                    n = name
                    s = s[n]

            descend(id.type)
            descend(id.perspective)
            descend(id.offset)
            descend(id.dist)
            descend(id.dir)

            if id.array:
                for j in range(0, id.array_length):
                    p[n][j] = get_slice(id.scale)
            else:
                p[n] = get_slice(id.scale)

        return data

    def __str__(self):
        s = ''
        for id in self._ids:
            s += 'pred config: %s\n' % str(id)
        return s[:-1]
