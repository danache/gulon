import mxnet as mx
data_iter = mx.io.ImageRecordIter(
    path_imgrec="/home/dan/gulon/notebook/data/caltech.rec", # the target record file
    data_shape=(3, 227, 227), # output data shape. An 227x227 region will be cropped from the original image.
    batch_size=4, # number of samples per batch
    resize=256 # resize the shorter edge to 256 before cropping
    # ... you can add more augumentation options as defined in ImageRecordIter.
    )
data_iter.reset()
batch = data_iter.next()

data =  mx.sym.Variable('data')
net = mx.symbol.Convolution(data=data, num_filter=64, kernel=7, stride=1, pad=3,name="conv1")
net = mx.symbol.FullyConnected(data=net,num_hidden=46,name='fc')
net = mx.sym.SoftmaxOutput(data=net, name='softmax')
mod = mx.mod.Module(symbol=net,
                        context=mx.cpu(),
                        label_names=('label',)
                        )
mod.bind(data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label)
mod.init_params(initializer= mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) )
# net = mx.symbol.FullyConnected(data=net,num_hidden=46,name='fc')
# net = mx.sym.SoftmaxOutput(data=net, name='softmax')