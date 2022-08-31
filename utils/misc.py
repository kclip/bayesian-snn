import fnmatch
import math
import os
import time
import numpy as np
from torch.nn.init import _calculate_correct_fan, calculate_gain



def make_experiment_dir(path, exp_type):
    prelist = np.sort(fnmatch.filter(os.listdir(path), '[0-9][0-9][0-9]__*'))
    if len(prelist) == 0:
        expDirN = "001"
    else:
        expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

    results_path = time.strftime(path + r'/' + expDirN + "__" + "%d-%m-%Y",
                                 time.localtime()) + '_' + exp_type

    os.makedirs(results_path)

    return results_path


def get_output_shape(input_shape, kernel_size=[3, 3],
                     stride=[1, 1], padding=[1, 1], dilation=[0, 0]):
    if not hasattr(kernel_size, '__len__'):
        kernel_size = [kernel_size, kernel_size]
    if not hasattr(stride, '__len__'):
        stride = [stride, stride]
    if not hasattr(padding, '__len__'):
        padding = [padding, padding]
    if not hasattr(dilation, '__len__'):
        dilation = [dilation, dilation]
    im_height = input_shape[-2]
    im_width = input_shape[-1]
    height = int((im_height + 2 * padding[0] - dilation[0] *
                  (kernel_size[0] - 1) - 1) // stride[0] + 1)
    width = int((im_width + 2 * padding[1] - dilation[1] *
                 (kernel_size[1] - 1) - 1) // stride[1] + 1)
    return [height, width]


def calculate_fan_in(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in cannot be computed "
                         "for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size

    return fan_in


def get_scale(layer, scaling=True):
    if scaling and hasattr(layer, 'weight'):
        fan = _calculate_correct_fan(layer.weight, mode='fan_in')
        gain = calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
        std = gain / math.sqrt(fan)
        scale = math.sqrt(3.0) * std
        # scale = scale ** 2
    else:
        scale = 1.

    return scale
