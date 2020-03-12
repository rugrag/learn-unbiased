import numpy as np
import os


class Digit_loader():

    def __init__(self, data_dir, config):

        var = config.var
        res = config.dim_c

        filename = 'mnist_10color_jitter_var_{}.npy'.format(var)
        filepath = os.path.join(data_dir, filename)

        data = np.load(filepath, encoding='latin1').item()

        self.x_train = data['train_image']
        self.y_train = data['train_label']
        self.x_test = data['test_image']
        self.y_test = data['test_label']

        self.y_train = self.convert_2_one_hot(self.y_train)
        self.y_test = self.convert_2_one_hot(self.y_test)

        __, inds = self.quantize_imgs(self.x_train, res)
        self.c_train = self.binary_c_lab(inds, res)

        __, inds = self.quantize_imgs(self.x_test, res)
        self.c_test = self.binary_c_lab(inds, res)

        self.N_samples = len(self.x_train)

    def next_batch(self, idx_batch):

        yield self.x_train[idx_batch], self.y_train[idx_batch], self.c_train[idx_batch]

    def convert_2_one_hot(self, y_raw):

        return np.eye(10)[y_raw]

    # TODO METTERE COMMENTI
    def quantize_imgs(self, imgs, bin_num=8):
        # sub-sample image

        N = imgs.shape[0]
        ss = imgs.reshape((N, 784, 3))
        q = np.amax(ss, axis=1)
        imgs_red = q.reshape((N, 1, 1, 3))

        # quantize colors
        step = 256 // bin_num
        bins = np.array(range(0, 255, step))

        inds = np.digitize(imgs_red, bins) - 1
        imgs_qnt = bins[inds]

        return imgs_qnt, inds

    # TODO METTERE COMMENTI
    def binary_c_lab(self, imgs, bin_num=8):
        N_samples = imgs.shape[0]

        d_label = imgs.reshape((N_samples, -1))
        N_pixels = d_label.shape[1]

        d_ext = np.zeros((N_samples, N_pixels * bin_num))

        for n in range(N_samples):
            l = []
            for i in range(N_pixels):
                a = np.zeros(bin_num)
                a[d_label[n, i]] = 1
                l.append(a)

            d_ext[n] = np.concatenate(l)

        return d_ext
