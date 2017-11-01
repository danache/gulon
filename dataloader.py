
import mxnet as mx
import iterator
data_iter = iterator.hgIter(
  path_imgrec="/home/dan/test_img/train.rec", # The target record file.
  # Output data shape; 227x227 region will be cropped from the original image.
  data_shape=(3, 256, 256),

    batch_size=1, # Number of items per batch.
max_rotate_angle=30,
rand_mirror=False,
  # You can specify more augmentation options. Use help(mx.io.ImageRecordIter) to see all the options.
)