import mxnet as mx
import numpy as np
import cv2
import PIL
from PIL import Image
import random
from matplotlib import pyplot as plt
import opt
class hgIter(mx.io.DataIter):
    """
    user for generate iter including heatmap
    """

    def __init__(self,path_imgrec,data_shape,batch_size,max_rotate_angle,mean_pixels=[0,0,0],rand_mirror=True,color_jitter=20):
        super(hgIter, self).__init__()
        self.rec = mx.io.ImageRecordIter(
            path_imgrec=path_imgrec,  # The target record file.
            # Output data shape; 227x227 region will be cropped from the original image.
            label_width=46,
            data_shape=data_shape,
            batch_size=batch_size,  # Number of items per batch.
            #rand_mirror=rand_mirror,
            random_h=color_jitter,
            random_s=color_jitter,
            random_l=color_jitter,
            mean_r = mean_pixels[0],
            mean_g=mean_pixels[1],
            mean_b=mean_pixels[2],

            # You can specify more augmentation options. Use help(mx.io.ImageRecordIter) to see all the options.
        )
        self.batch_size = batch_size
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)
        self.reset()

    @property
    def provide_data(self):
        return self.rec.provide_data  # [('data', (self.batch_size, 3, 256, 256))]

    def reset(self):
        self.rec.reset()

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False
        #data = mx.nd.cast(self._batch.data[0],dtype='uint8').asnumpy()
        #data = np.transpose(data,(2,3,1,0))
        #img_data = np.reshape(data, (256,256,3))
        #img_data = cv2.fromarray(img_data)
        #imgcp = img_data.copy()
        heatmap = np.zeros((self._batch.label[0].shape[0],opt.partnum,opt.outputRes, opt.outputRes ), dtype=np.float32)

        for i in range(self._batch.label[0].shape[0]):
            label = self._batch.label[0][i].asnumpy()
            for j in range(opt.partnum):
                x = 4 + j * 3 + 1
                y = 4 + j * 3 + 2
                #imgcp = cv2.circle(imgcp, (int(label[x]*256), int(label[y]*256)), 15, (0, 0, 255), -1)
                s = int(np.sqrt(opt.outputRes) * opt.outputRes * 10 / 4096) + 2
                hm = self._makeGaussian(opt.outputRes, opt.outputRes, sigma=s, center=(label[x]*opt.outputRes, label[y]*opt.outputRes))
                heatmap[i,j,:, :] = hm
                #hm = hm.astype(np.uint8)

        y_batch = np.zeros((self.batch_size, opt.nStack, opt.partnum,opt.outputRes, opt.outputRes ))
        for i in range(opt.nStack):
            y_batch[:,i,:,:,:] = heatmap
        self.label_shape = (self.batch_size, opt.nStack, opt.partnum, opt.outputRes, opt.outputRes)
        self.provide_label = [('hg_label', self.label_shape)]
        #cv2.imwrite("/home/dan/test_img/2222.jpg", imgcp)
        self._batch.label = [mx.nd.array(y_batch)]
        #self.showHeatMap()
        return True


    def showHeatMap(self):
        for i in range(self.heatmap.shape[0]):
            fig = plt.figure()
            for j in range(14):
                ax = fig.add_subplot(4,4,j+1)
                img = self.heatmap[i,:,:,j]
                ax.imshow(img)
            plt.show()

    # def _augment(self, img, hm, max_rotation=30):
    #     """ # TODO : IMPLEMENT DATA AUGMENTATION
    #     """
    #     if random.choice([1]):
    #         r_angle = np.random.randint(-1 * max_rotation, max_rotation)
    #         img = transform.rotate(img, r_angle, preserve_range=True)
    #         hm = transform.rotate(hm, r_angle)
    #     return img, hm


                # img = Image.fromarray(data)
        # img.save("/home/dan/text_img/ssss.png")
        #for i in range(14):

        #self.heatmap = self._generate_hm(256, 256, 14, 256)

    def _makeGaussian(self, height, width, sigma=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def _generate_hm(self, height, width, joints, maxlenght):
        """ Generate a full Heap Map for every joints in an array
        Args:
            height			: Wanted Height for the Heat Map
            width			: Wanted Width for the Heat Map
            joints			: Array of Joints
            maxlenght		: Lenght of the Bounding Box
        """
        num_joints = joints.shape[0]
        hm = np.zeros((height, width, num_joints), dtype=np.float32)
        for i in range(num_joints):
            if not (np.array_equal(joints[i], [-1, -1])):
                s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
                hm[:, :, i] = self._makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm