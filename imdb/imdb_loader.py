import numpy as np
import numpy.random as npr
import os

from PIL import Image


class Imdb_loader():

    def __init__(self, config, data_dir, exp, n_samples):

        # TODO TOGLIERE
        data_dir = '/data/rvolpi/imdb'

        self.img_path = os.path.join(data_dir, 'imdb_crop')
        self.config = config

        # age binning according to Alvi et al.
        self.bins = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120])

        print(self.img_path)

        # images
        self.tr_imgs, self.tr_gt_dict = self.get_split(exp, split='train_list')
        self.tr_imgs = np.array(self.tr_imgs)

        # generate ground truth
        self.tr_gt = np.zeros((len(self.tr_imgs), 2))  # first column age, second column gender

        for n, tr_img in enumerate(self.tr_imgs):
            self.tr_gt[n, 0] = self.tr_gt_dict[tr_img.encode('utf-8')]['age']
            self.tr_gt[n, 1] = self.tr_gt_dict[tr_img.encode('utf-8')]['gender']


        # shuffling - not really necessary here
        indices = np.arange(len(self.tr_imgs), dtype=int)
        npr.shuffle(indices)
        self.tr_imgs = self.tr_imgs[indices[:]]
        self.tr_gt = self.tr_gt[indices[:]]

        if n_samples != -1:
            print('\n\n\nWARNING - USING FEW SAMPLES\n\n\n')

        # select sub-set of image split
        self.tr_imgs = self.tr_imgs[:n_samples]
        self.tr_gt = self.tr_gt[:n_samples]

        # ------ Test data ------
        self.ts_imgs, self.ts_gt_dict = self.get_split(exp, split='test_list')
        self.ts_imgs = np.array(self.ts_imgs)

        # generate test data ground truth
        self.ts_gt = np.zeros((len(self.ts_imgs), 2))  # first column age, second column gender

        for n, ts_img in enumerate(self.ts_imgs):
            self.ts_gt[n, 0] = self.ts_gt_dict[ts_img.encode('utf-8')]['age']
            self.ts_gt[n, 1] = self.ts_gt_dict[ts_img.encode('utf-8')]['gender']

        # shuffling - not really necessary here
        indices = np.arange(len(self.ts_imgs), dtype=int)
        npr.shuffle(indices)
        self.ts_imgs = self.ts_imgs[indices[:]]
        self.ts_gt = self.ts_gt[indices[:]]

        self.N_samples = len(self.tr_imgs)



    def next_batch(self, idx_batch, split='train'):

        if split == 'train':
            imgs_batch, gender_lb, age_lb = self.extract_batch(self.tr_imgs, self.tr_gt, idx_batch)
        if split == 'test':
            imgs_batch, gender_lb, age_lb = self.extract_batch(self.ts_imgs, self.ts_gt, idx_batch)

        yield imgs_batch, gender_lb, age_lb


    def extract_batch(self, imgs, gt, idx_batch):


        imgs_batch = np.zeros((self.config.batch_size, 224, 224, 3), dtype=np.float64)
        age_lb = np.zeros(self.config.batch_size, dtype=int)
        gender_lb = np.zeros(self.config.batch_size, dtype=int)

        for i, idx in enumerate(idx_batch):

            #image file name to be loaded
            img_name = imgs[idx]

            # load img_name
            try:
                img = Image.open(os.path.join(self.img_path, img_name))
            except:
                # marking missing images so we can remove them later
                age_lb[i] = -1000
                continue

            age_lb[i] = gt[idx][0]
            gender_lb[i] = gt[idx][1]

            img = img.resize((224, 224), Image.ANTIALIAS)
            img = np.expand_dims(img, axis=0)

            # if grayscale image add channels
            if img.shape == (1, 224, 224):
                img = np.repeat(img[:, :, :, np.newaxis], 3, axis=3)

            # create images vector
            imgs_batch[i] = img

        # removing missing images and associated annotations
        imgs_batch = imgs_batch[age_lb != -1000]
        gender_lb = gender_lb[age_lb != -1000]
        age_lb = age_lb[age_lb != -1000]

        # ImageNet data normalization
        imgs_batch[:, :, :, 0] -= 103.939
        imgs_batch[:, :, :, 1] -= 116.779
        imgs_batch[:, :, :, 2] -= 123.68

        # fai i bin di age_lb
        age_lb = self.quantize_age(age_lb, self.bins)

        # one-hot gender_lb
        gender = np.zeros((self.config.batch_size, 2))
        for i in range(self.config.batch_size):
            gender[i, gender_lb[i]] = 1


        return imgs_batch, gender, age_lb

    # retrieve filenames for each split
    def get_split(self, exp, split='train_list'):

        gt = np.load(os.path.join(self.img_path, 'imdb_age_gender.npy'), allow_pickle=True, encoding='latin1').item()

        if self.config.exp_name == 'base' or split == 'test_list':
            sp = np.load(os.path.join(self.img_path, 'imdb_split.npy'), allow_pickle=True, encoding='latin1').item()
            image_list = sp[split]  # depends on whether train or test (train_list/test_list)

        if exp == 'eb1' and split == 'train_list':
            image_list = np.load(os.path.join(self.img_path, 'eb1_img_list.npy'), allow_pickle=True, encoding='latin1')

        if exp == 'eb2' and split == 'train_list':
            image_list = np.load(os.path.join(self.img_path, 'eb2_img_list.npy'), allow_pickle=True, encoding='latin1')

        return image_list, gt

    # get binary vector for age
    def quantize_age(self, age_vec, bins):
        n_samples = age_vec.shape[0]
        n_bins = bins.shape[0] - 1

        age_lb = np.zeros((n_samples, n_bins))
        hh = np.digitize(age_vec, bins) - 1

        # TODO remove for loop somehow
        for i in range(n_samples):
            age_lb[i, hh[i]] = 1

        return age_lb
