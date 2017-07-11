import os
import sys
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
from keras import losses
from keras import backend as K

_EPSILON = K.epsilon()

def _loss_tensor_example(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

def _loss_np_example(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)

def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)

    tp = K.sum(K.round(y_true * y_pred)) + _EPSILON
    fp = K.sum(K.round(K.clip(y_pred - y_true, _EPSILON, 1.0-_EPSILON)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, _EPSILON, 1.0-_EPSILON)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    out = 1.0 - ( (precision * recall) / (precision + recall + _EPSILON) )

    return out

def _loss_np(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)

    tp = np.sum(np.round(y_true * y_pred)) + _EPSILON
    fp = np.sum(np.round(np.clip(y_pred - y_true, _EPSILON, 1.0-_EPSILON)))
    fn = np.sum(np.round(np.clip(y_true - y_pred, _EPSILON, 1.0-_EPSILON)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    out = 1.0 - ( (precision * recall) / (precision + recall + _EPSILON) )

    return out


def check_loss(_shape):
    if _shape == '2d':
        shape = (6, 7)
    elif _shape == '3d':
        shape = (5, 6, 7)
    elif _shape == '4d':
        shape = (8, 5, 6, 7)
    elif _shape == '5d':
        shape = (9, 8, 5, 6, 7)

    y_a = np.random.random(shape)
    y_b = np.random.random(shape)

    out1 = K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))
    out2 = _loss_np(y_a, y_b)

    print(out1)
    print(out2)
    print(out1.shape)
    print(out2.shape)

    assert(out1.shape == out2.shape)
    assert(out1.shape == shape[:-1])
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))


def test_loss():
    shape_list = ['2d', '3d', '4d', '5d']
    for _shape in shape_list:
        check_loss(_shape)
        print('======================')

if __name__ == '__main__':
    test_loss()
