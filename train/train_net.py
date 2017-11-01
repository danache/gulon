import argparse
#import tools.find_mxnet
import mxnet as mx
import os
import sys
import logging
import opt

from model.hg_attention import createModel
from iterator import  hgIter
def train_net():
    ctx = [mx.gpu(int(i)) for i in opt.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file = "train.log"
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    train_iter = hgIter(path_imgrec=opt.train_path, data_shape=opt.data_shape, batch_size=opt.batch_size,max_rotate_angle=30,
                       mean_pixels=opt.mean_pixels,color_jitter=20)
    # train_iter = mx.io.ImageRecordIter(path_imgrec=opt.train_path,  # The target record file.
    #         # Output data shape; 227x227 region will be cropped from the original image.
    #         label_width=46,
    #         data_shape=opt.data_shape,
    #         batch_size=opt.batch_size
    #                                    ) # Number of items per batch.
    # if opt.val_path:
    #     val_iter = hgIter(path_imgrec=opt.val_path, data_shape=opt.data_shape, batch_size=opt.batch_size,max_rotate_angle=30,
    #                     mean_pixels=opt.mean_pixels,color_jitter=20)
    # else:
    #     val_iter = None
    #print(train_iter._batch.label)

    data =  mx.sym.Variable('data')
    net = createModel(data)
    mod = mx.mod.Module(symbol=net,
                        context=ctx,
                        logger=logger,
                        label_names=('hg_label',)
                        )

    optimizer_params = {'learning_rate': 2.5e-4}
    mod.fit(train_iter,
            train_iter,
            batch_end_callback=mx.callback.Speedometer(train_iter.batch_size, 1),
            epoch_end_callback=mx.callback.do_checkpoint("model/hg/",period=100),
            optimizer="RMSProp",
            optimizer_params=optimizer_params,
            begin_epoch=1,
            num_epoch=200,
            initializer=mx.init.Xavier()
            )