import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os
import time
import math
import numpy as np
import my_txtutils as txt

SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = txt.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001
dropout_pkeep = 0.8  # some dropout

# load data, either shakespeare, or the Python source of Tensorflow itself
# shakedir = "shakespeare/*.txt"
shakedir = "shakespeare/hamlet.txt"
# shakedir = "../tensorflow/**/*.py"
codetext, valitext, bookranges = txt.read_data_files(shakedir, validation=True)

# display some stats on the data
epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
txt.print_data_stats(len(codetext), len(valitext), epoch_size)

