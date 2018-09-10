

SUCCESS = 0
STOPPED_EARLY = 3
DATA_PRESENT = 10
DATA_MISSING = 11
CONFIGURATION_ERROR = 12

def to_string(s):
    for k in globals():
        v = globals()[k]
        if isinstance(v, int):
            if v == s:
                return k