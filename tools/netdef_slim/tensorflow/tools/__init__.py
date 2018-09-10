import numpy as np


def image_to_tf_batch(img, num=1, normalize=True):
    normalizer = 255.0 if normalize else 1.0

    return np.tile(np.expand_dims(np.transpose(img, [2, 0, 1]), 0), [num, 1, 1, 1]).astype(np.float32)/normalizer

def tf_to_image(tensor, unnormalize=True):
    unnormalizer = 255.0 if unnormalize else 1.0

    return (np.transpose(tensor, [1, 2, 0])*unnormalizer).astype(np.uint8)

def flow_to_tf_batch(img, num=1):
    return np.tile(np.expand_dims(np.transpose(img, [2, 0, 1]), 0), [num, 1, 1, 1]).astype(np.float32)

def tf_to_flow(tensor):
    return (np.transpose(tensor, [1, 2, 0])).astype(np.float32)
