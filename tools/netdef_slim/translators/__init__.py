import netdef_slim as nd


def softmax2_soft_translator(data):
    return nd.ops.slice(nd.ops.softmax(data), 1)[1]

def softmax2_hard_translator(data):
    return nd.ops.slice(nd.ops.threshold(nd.ops.softmax(data), 0.5), 1)[1]
