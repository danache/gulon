from model.layers.AttentionPartsCRF import *
from model.layers.ResidualPool import *
from model.layers.Residual import *
from mxnet import gluon
from mxnet import symbol
import opt
from model.layers.Residual import *

def test_model(data):
    return AttentionPartsCRF(data, opt.nFeats,opt.LRNKer, 3, 0, name="hourglass_attention1")

data = mx.symbol.Variable('data')
conv_comp= test_model(data)

shape= {"data" : (2,256,64,64)}
mx.viz.plot_network(symbol=conv_comp, shape=shape).view()