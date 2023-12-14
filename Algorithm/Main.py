import sys
sys.path.append("../")

import os
import numpy as np
import random
from scipy.optimize import basinhopping
import tensorflow as tf
from tensorflow.python.platform import flags
import copy





if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_integer('sens_param', 9, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_string('model_path', '../models/census/test.Algorithm', 'the path for testing Algorithm')
    flags.DEFINE_integer('max_global', 1000, 'number of maximum samples for global search')
    flags.DEFINE_integer('max_local', 1000, 'number of maximum samples for local search')
    flags.DEFINE_float('step_size', 1.0, 'step size for perturbation')

    tf.app.run()