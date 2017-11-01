from model.layers.ResidualPool import *
import mxnet
import numpy as np
from mxnet import nd, autograd
from model.layers.AttentionPartsCRF import *

def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype(np.float32)
    return data, label

ctx = mx.gpu()
batch_size = 1
train_data = mx.gluon.data.DataLoader(mxnet.gluon.data.vision.ImageFolderDataset
                                      ("/home/dan/test_img",transform=transformer),batch_size=1)

for d, l in train_data:
    break
print(d.shape, l.shape)

#et = ResidualPool(64)
from model.layers.AttentionPartsCRF import *
from model.hg_attention import *
#net = AttentionPartsCRF(1,4,1)
net = createModel( 4, 8, 256,64,1, )
print(net)
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)



softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


def evaluate_accuracy(data_iterator, net):

    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

def train(net):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})
    #print(trainer._params[0].values())
    epochs = 10
    smoothing_constant = .01

    for e in range(epochs):
        for i, (d, l) in enumerate(train_data):
            data = d.as_in_context(ctx)
            label = l.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)

            loss.backward()
            trainer.step(data.shape[0])

            ##########################
            #  Keep a moving average of the losses
            ##########################
            curr_loss = nd.mean(loss).asscalar()

        train_accuracy = evaluate_accuracy(train_data, net)
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        print("Epoch %s. Loss: %s, Train_acc %s" % (e, moving_loss, train_accuracy))

train(net)