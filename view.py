import mxnet as mx
import numpy as np
import cv2
import matplotlib.pyplot as plt
import model.layers.Residual
import logging
from model.hg_attention import createModel
# 启动日志
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 定义一个网络
data = mx.symbol.Variable('data')
conv_comp= createModel(data)

#shape= {"data" : (2,14,256,256)}
shape= {"data" : (2,3,256,256)}
#conv_comp= ConvFactory(data=data, num_filter=64, kernel=(7,7), stride=(2,2))
#shape= {"data" : (128,3,28,28)}
mx.viz.plot_network(symbol=conv_comp, shape=shape).view()
#mx.viz.plot_network(symbol=conv_comp).view()