import mxnet as mx
import numpy as np
from mxnet import gluon
import opt
import copy
def AttentionIter(data, numin,lrnSize, iterSize,name=""):

    pad = np.int(np.floor(lrnSize / 2))
    U = mx.symbol.Convolution(data=data, num_filter=1, kernel=(3,3), stride=(1,1),pad=(1,1),
                                  name='%s_U'%(name) )
    w = mx.sym.Variable('%s_spConv_weight'%(name))
    b = mx.sym.Variable('%s_spConv_bias'%(name))
    spConv = mx.symbol.Convolution(data=U, weight = w, bias = b,num_filter=1, kernel=(lrnSize,lrnSize), stride=(1,1),pad=(pad,pad),
                                  name='%s_spConv' %(name))
    add1 = mx.symbol.add_n(U, spConv,name="%s_add_layer_1"%(name))
    sigmod1 =  mx.symbol.Activation(data=add1, act_type='sigmoid', name='%s_sigmod_0'%(name) )

    c = [spConv]
    ac = [sigmod1]


    for i in range(1,iterSize):
        conv =  mx.symbol.Convolution(data=ac[i - 1], weight = w,bias=b, num_filter=1, kernel=(lrnSize,lrnSize), stride=(1,1),pad=(pad,pad),
                                  name='%s_spConv_clone_%d' % (name,i))
        addn = mx.symbol.add_n(U, conv,name="%s_add_%d"%(name,i))
        sigmod_tmp = mx.symbol.Activation(data=addn, act_type='sigmoid', name='%s_sigmod_%d' % (name,i))
        ac.append(sigmod_tmp)
    tmp = ac[iterSize - 1]

    tmp2 = mx.symbol.tile(tmp, reps=(1, numin, 1, 1),name="%s_tile"%(name))
    x = mx.symbol.broadcast_mul(data, tmp2,name="%s_mul"%(name))
    return x
def AttentionPartsCRF(data,numin,lrnSize, iterSize, usepart,name=""):

    if usepart == 0:

        return AttentionIter(data,numin,lrnSize,iterSize=iterSize,name="%s_Attention"%(name))
    else:
        partnum = opt.partnum#opt.partnum
        pre = []
        for i in range(partnum):
            att = AttentionIter(data=data,numin=numin,lrnSize=lrnSize, iterSize=iterSize,name="%s_Attention_%d"%(name,i))
            tmpconv = mx.symbol.Convolution(data=att,  num_filter=1, kernel=(1,1), stride=(1,1),
                                  name='%s_conv_%d' % (name,i))
            pre.append(tmpconv)
        # res = pre[0]
        # for i in range(1,partnum):

        res = mx.symbol.concat(*pre,dim=1,name="%s_concat_%d"%(name, i))
        return res
        #return mx.symbol.concat(data=pre, dim=-1)

#
# def replicate(self, input, numIn, dim, name):
#         repeat = []
#         for i in range(numIn):
#             repeat.append(input)
#         return mx.symbol.concat(repeat, dim)

"""
class AttentionIter(gluon.HybridBlock):
    def __init__(self, lrnSize, iterSize, **kwargs):
        super(AttentionIter, self).__init__(**kwargs)
        self.pad = np.int(np.floor(lrnSize / 2))
        self.iterSize = iterSize
        self.lrnSize = lrnSize
        with self.name_scope():
            self.U = gluon.nn.Conv2D(channels=1,kernel_size=3,strides=1,padding=1)
            self.spConv = gluon.nn.Conv2D(1,kernel_size=lrnSize,strides=1,padding=(self.pad))
            self.C = []
            self.ac = []
            self.C.append(self.spConv)
            self.sig1 = gluon.nn.Activation(activation="sigmoid")
            self.ac.append(self.sig1)
            for i in range(1,iterSize):
                tmpcv = gluon.nn.Conv2D(1,kernel_size=lrnSize,strides=1,padding=self.pad,params=self.spConv.params)
                tmprelu = gluon.nn.Activation(activation="sigmoid")
                self.register_child(tmpcv)
                self.register_child(tmprelu)
                self.C.append(tmpcv)
                self.ac.append(tmprelu)


    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym

        Q = []
        lin_x = self.U(x)
        U = lin_x

        for i in range(self.iterSize):


            lin_x = self.C[i](lin_x)
            q_sum =F.add_n(lin_x, U)
            #q_sum = F.broadcast_add(lin_x, U)
            lin_x = self.ac[i](q_sum)
            Q.append(lin_x)
    #################
    #######question here
    #################
        tmp = Q[self.iterSize - 1]
        tmp2 =  F.tile(tmp, reps=(1,x.shape[1],1,1))
        x = F.broadcast_mul(x, tmp2)
        return x
    ##############################

class AttentionPartsCRF(gluon.HybridBlock):
    def __init__(self, lrnSize, iterSize, usepart, **kwargs):
        super(AttentionPartsCRF, self).__init__(**kwargs)

        self.lrnSize = lrnSize
        self.iterSize = iterSize
        self.usepart = usepart
        with self.name_scope():

            self.att = AttentionIter(lrnSize=self.lrnSize,iterSize=self.iterSize)

            self.att2 = []
            self.conv = []
            for i in range(14):
                tmp = AttentionIter(lrnSize=self.lrnSize,iterSize=self.iterSize)
                self.register_child(tmp)
                self.att2.append(tmp)
                tmpconv = gluon.nn.Conv2D(1,kernel_size=1,strides=1,padding=0)
                self.register_child(tmpconv)
                self.conv.append(tmpconv)



    def hybrid_forward(self, F, x):
        if self.usepart == 0:
            return self.att(x)
        else:
            partnum = 14 ##### should be dynamic
            pre = []
            for i in range(partnum ):
                x = self.att2[i](x)
                x = self.conv[i](x)
                pre.append(x)
            x = pre[0]
            for i in range(1,len(pre)):
                x = F.concat(x,pre[i],dim=1)
        return x
"""