from model.layers.AttentionPartsCRF import *
from model.layers.ResidualPool import *
from model.layers.Residual import *
from mxnet import gluon
from mxnet import symbol
import opt
from model.layers.Residual import *

def repResidual(data, num, nRep,name=None,suffix=""):
    #with mx.name.Prefix("%s_%s_" % (name, suffix)):
    out = []

    for i in range(nRep):
        tmpout = None
        if i == 0:
            tmpout = Residual(data,numin=num,numOut=num,name="%s_tmp_out%d"%(name,i))
        else:
            tmpout = ResidualPool(data=out[i - 1],numin=num,numOut=num,name="%s_tmp_out%d"%(name, i))
        out.append(tmpout)

    return out[-1]

def hourglass(data,n, f, imsize, nModual,name="",suffix=""):
    #with mx.name.Prefix("%s_%s_" % (name, suffix)):
    pool =  mx.symbol.Pooling(data=data, kernel=(2,2),stride=(2,2),pool_type="max",name='%s_pool1' %(name) )

    up = []
    low = []

    for i in range(nModual):

        tmpup = None
        tmplow = None
        if i == 0:
            if n > 1:
                tmpup = repResidual(data,num=f,nRep=n-1,name='%s_tmpup_'%(name) + str(i))
            else:
                tmpup = Residual(data,f,f,name='%s_tmpup_'%(name) + str(i))
            tmplow = Residual(pool,f,f,name='%s_tmplow_'%(name) + str(i))
        else:
            if n > 1:
                tmpup = repResidual(up[i-1],f, n - 1,name='%s_tmpup_'%(name) + str(i))
            else:
                tmpup = ResidualPool(up[i-1],f,f,name='%s_tmpup_'%(name) + str(i))
            tmplow = Residual(low[i - 1],f,f,name='%s_tmplow_'%(name) + str(i))

        up.append(tmpup)
        low.append(tmplow)

    ####################
    #####################

    low2 = None
    if n > 1:
        low2 = hourglass(low[nModual - 1],n - 1,f,imsize / 2,nModual=nModual,name=name+"_" + str(n - 1)+"_low2")
    else:
        low2 = Residual(low[nModual - 1], f,f,name=name+"_"+str(n - 1)+"_low2")
    low3 = Residual(low2, f,f,name=name+"_"+str(n)+"low3")
    up2 = mx.symbol.UpSampling(low3, scale=2, sample_type='nearest',name="%s_Upsample"%(name))


    comb = mx.symbol.add_n(up[nModual - 1], up2,name=name+"_add")
    return comb


def lin(data, numOut,name=None, suffix=""):
    with mx.name.Prefix("%s_%s_" % (name, suffix)):

        conv1 = mx.symbol.Convolution(data=data, num_filter=numOut, kernel=(1,1), stride=(1,1),
                                      name='conv1' )
        bn1 = mx.symbol.BatchNorm(data=conv1, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9,
                                  name='bn1' )

        relu1 = mx.symbol.Activation(data=bn1, act_type='relu', name='relu1' )
        return relu1

def createModel(data):
    label = mx.symbol.Variable(name="hg_label")
    conv1 = mx.symbol.Convolution(data=data, num_filter=64, kernel=(7,7), stride=(1,1), pad=(3,3),name="conv1")


    bn1 = mx.symbol.BatchNorm(data=conv1, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9,name="bn1")

    relu1 = mx.symbol.Activation(data=bn1, act_type='relu',name="relu1")
    r1 = Residual(relu1, 64,64,name="Residual1")

    pool1 = mx.symbol.Pooling(data=r1, kernel=(2,2),stride=(2,2),pool_type="max",name="pool1")
    r2 = Residual(pool1, 64,64,name="Residual2")

    r3 = Residual(r2, 64,128,name="Residual3")

    pool2 = mx.symbol.Pooling(data=r3, kernel=(2,2),stride=(2,2), pool_type="max",name="pool2")
    r4 = Residual(pool2, 128,128,name="Residual4")

    r5 = Residual(r4, 128,128,name="Residual5")

    r6 = Residual(r5,128,opt.nFeats,name="Residual6")

    ####################################################
    #r6 = data
    ####################################################
    out = []
    inter = [r6]
    nPool = opt.nPool
    if nPool == 3:
        nModual = 16 / opt.nStack
    else:
        nModual = 8 / opt.nStack
    ###################################################test



    ##########################################################3
    for i in range(opt.nStack):

        hg = hourglass(data=inter[i],n=nPool,f=opt.nFeats,imsize=opt.outputRes,nModual=int(nModual),name="hourglass%d" %(i))#n=nPool,f=opt.nFeats,imsize=opt.outputRes,nModual=int(nModual))

        ll1 = None
        ll2 = None
        att = None
        tmpOut = None
        if i == opt.nStack - 1:
            ll1 = lin(hg,opt.nFeats * 2,name="hourglass%d_lin1"%(i))
            ll2 = lin(ll1,opt.nFeats * 2,name="hourglass%d_lin2"%(i))
            att = AttentionPartsCRF(ll2, opt.nFeats * 2,opt.LRNKer, 3, 0,name="hourglass%d_attention1"%(i))
            tmpOut = AttentionPartsCRF(att, opt.nFeats * 2,opt.LRNKer, 3, 1,name="hourglass%d_attention2"%(i))
        else:
            ll1 = lin(hg, opt.nFeats,name="hourglass%d_lin1"%(i))
            ll2 = lin(ll1, opt.nFeats,name="hourglass%d_lin2"%(i))

            if i >= 4 :
                att = AttentionPartsCRF(ll2, opt.nFeats,opt.LRNKer, 3, 0,name="hourglass%d_attention1"%(i))
                tmpOut = AttentionPartsCRF(att, opt.nFeats,opt.LRNKer, 3, 1,name="hourglass%d_attention2"%(i))
            else:

                att = AttentionPartsCRF(ll2, opt.nFeats,opt.LRNKer, 3, 0,name="hourglass%d_attention1"%(i))

                tmpOut = mx.symbol.Convolution(data=att, num_filter=opt.partnum, kernel=(1,1), stride=(1,1), pad=0,name="hourglass%d_tmpout"%(i))

        out.append(tmpOut)

        if i < opt.nStack - 1:
            outmap = mx.symbol.Convolution(data=tmpOut, num_filter=256, kernel=(1,1), stride=(1,1), pad=0,name="hourglass%d_conv"%(i))
            ll3 = lin(outmap, opt.nFeats,name="hourglass%d_lin3"%(i))
            toointer = mx.symbol.add_n(inter[i],outmap,ll3,name="add_n%d"%(i))
            inter.append(toointer)

    arg_shapes, out_shapes, aux_shapes = out[0].infer_shape(data=(1, 3, 256, 256))
    print(out_shapes)


    for i in range(len(out)):
        out[i] = mx.symbol.expand_dims(out[i],axis=1)
    loss = mx.symbol.concat(*out,dim=1 ,name="loss")
    arg_shapes, out_shapes, aux_shapes = loss.infer_shape(data=(1, 3, 256, 256))
    print(out_shapes)
    #return out[opt.nStack - 1]
    #group = mx.sym.Group(out)
    loss = mx.symbol.LinearRegressionOutput(data=loss,label=label,name="hg")

    return loss




"""
class repResidual(gluon.HybridBlock):
    def __init__(self,  num, nRep, **kwargs):
        super(repResidual, self).__init__(**kwargs)
        self.nRep = nRep
        with self.name_scope():
            self.out = []
            for i in range(nRep):
                tmpout = None
                if (i == 0):
                    tmpout = Residual(num)
                else:
                    tmpout = ResidualPool(num)
                self.register_child(tmpout)
                self.out.append(tmpout)

                #     exec("tmpout%d = Residual(num)"%(i))
                # else:
                #     exec("tmpout%d = ResidualPool(num)" % (i))
                # exec ("self.register_child(tmpout%d)" % (i))
                # exec ("self.out.append(tmpout%d)" % (i))

    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym
        for i in range( self.nRep):
            x = self.out[i](x)
        return x

class hourglass(gluon.HybridBlock):
    def __init__(self, n, f, imsize, nModual, **kwargs):
        super(hourglass, self).__init__(**kwargs)
        with self.name_scope():
            self.nModual = nModual
            self.pool = gluon.nn.MaxPool2D(pool_size=2,strides=2)
            self.up = []
            self.low = []
            for i in range(int(nModual)):
                tmpup = None
                tmplow = None
                if i == 0:
                    if n > 1:
                        tmpup = repResidual(f,n - 1)
                    else:
                        tmpup = Residual(f)
                    tmplow = Residual(f)
                else:
                    if n > 1:
                        tmpup = repResidual(f,n - 1)
                    else:
                        tmpup = ResidualPool(f)
                    tmplow = Residual(f)
                self.register_child(tmpup)
                self.register_child(tmplow)
                self.up.append(tmpup)
                self.up.append(tmplow)

            if n > 1:
                self.low2 = hourglass(n-1,f,imsize/2,nModual)
            else:
                self.low2 = Residual(f)
            self.low3 = Residual(f)


    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym
        pool = self.pool(x)
        for i in range(int(self.nModual)):
            x = self.up[i](x)
            pool = self.low[i](x)

        pool = self.low2(pool)
        pool = self.low3(pool)
        up2 =  F.UpSampling(pool, scale=2, sample_type='nearest')
        comb = F.add_n(x, up2)
        return comb

class lin(gluon.HybridBlock):
    def __init__(self,  numOut,  **kwargs):
        super(lin, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(channels=numOut,kernel_size=1,strides=1,padding=0)
            self.bn1 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
            self.relu1 = gluon.nn.Activation(activation='relu')

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

class createModel(gluon.HybridBlock):
    def __init__(self,  nPool, nStack,nFeats,outputRes,LRNKer, **kwargs):
        super(createModel, self).__init__(**kwargs)
        self.nStack = nStack
        with self.name_scope():
            self.cnv1_ = gluon.nn.Conv2D(channels=64, kernel_size=7, strides=1, padding=3)
            # cnv1
            self.cnv1 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
            # relu1
            self.relu1 = gluon.nn.Activation(activation='relu')
            # r1
            self.r1 = Residual(64)
            # pool1
            self.pool1 = gluon.nn.MaxPool2D(pool_size=2, strides=2, )
            # r2
            self.r2 = Residual(64)
            # r3
            self.r3 = Residual(128)

            # pool2
            self.pool2 = gluon.nn.MaxPool2D(pool_size=2, strides=2, )
            # r4
            self.r4 = Residual(128)
            # r5
            self.r5 = Residual(128)
            # r6
            self.r6 = Residual(nFeats)

            self.out = []

            self.ll1_set = []
            self.ll2_set= []
            self.att1_set = []

            self.hg_set = []

            self.outmap_set = []
            self.ll3_set = []

            if nPool == 3:
                nModual = 16 / nStack
            else:
                nModual = 8 / nStack
            for i in range(nStack):
                hg = hourglass(nPool, nFeats, outputRes, nModual)
                self.register_child(hg)
                self.hg_set.append(hg)
                ll1 = None
                ll2 = None
                att = None
                tmpout = None
                if i == nStack - 1:
                    ll1 = lin(nFeats * 2)
                    ll2 = lin(nFeats * 2)
                    att = AttentionPartsCRF(LRNKer,3,0)
                    tmpout = AttentionPartsCRF(LRNKer,3,1)
                else:
                    ll1 = lin(nFeats)
                    ll2 = lin(nFeats)
                    if i >= 4:
                        att = AttentionPartsCRF(LRNKer, 3, 0)
                        tmpout = AttentionPartsCRF(LRNKer, 3, 1)
                    else:
                        att = AttentionPartsCRF(LRNKer, 3, 0)
                        tmpout = gluon.nn.Conv2D(channels=14, kernel_size=1, strides=1, padding=0)
                self.register_child(ll1)
                self.register_child(ll2)
                self.register_child(att)
                self.register_child(tmpout)

                self.ll1_set.append(ll1)
                self.ll2_set.append(ll2)
                self.att1_set.append(att)
                self.out.append(tmpout)


                if i < nStack - 1:
                    outmap = gluon.nn.Conv2D(256, strides=1, kernel_size=1, padding=0)
                    ll3 = lin(nFeats)
                    self.register_child(outmap)
                    self.register_child(ll3)
                    self.outmap_set.append(outmap)
                    self.ll3_set.append(ll3)

    def hybrid_forward(self, F, x):
        x = self.cnv1_(x)
        x = self.cnv1(x)
        x = self.r1(x)
        x = self.pool1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.pool2(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.r6(x)
        inter = x
        for i in range(self.nStack):
            hg = self.hg_set[i](x)
            ll1 = self.ll1_set[i](hg)
            ll2 = self.ll2_set[i](ll1)
            att = self.att1_set[i](ll2)
            tmpout = self.out[i](att)
            if i < self.nStack - 1 :
                outmap = self.outmap_set[i](tmpout)
                ll3 = self.ll3_set[i](ll1)
                tmpout = F.add_n(inter,outmap,ll3)
                inter = tmpout
            x = tmpout

        return x

"""