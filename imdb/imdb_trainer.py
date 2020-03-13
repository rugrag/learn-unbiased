import numpy.random as npr
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os


class Imdb_trainer():
    def __init__(self, sess, model, imdb_eb1, imdb_eb2, config, logger, data_dir):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.imdb_eb1 = imdb_eb1
        self.imdb_eb2 = imdb_eb2
        self.data_dir = data_dir
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        if   self.config.exp_name == 'eb1':
            print('train on EB1')
            self.data = imdb_eb1
        elif self.config.exp_name == 'eb2':
            print('train on EB2')
            self.data = imdb_eb2

        """
        Initializes ResNet-50 network parameters using slim.
        """

        # restoring only the layers up to logits (excluded)
        model_variables = slim.get_model_variables('resnet_v1_50')
        variables_to_restore = slim.filter_variables(model_variables,
                                                     exclude_patterns=['logits', 'bottleneck_layer', 'logit_layer'])
        # loading ResNet checkpoint
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(self.sess, os.path.join(self.data_dir, 'resnet_v1_50.ckpt'))

        self.index_list = range(self.data.N_samples)
        self.num_iter_per_epoch = self.data.N_samples // self.config.batch_size

        self.check = 5


    def train(self):
        """
        Trains the model defined via config file
        """
        self.step = 0.
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.cur_epoch = cur_epoch
            idx_permutation = npr.permutation(range(self.data.N_samples))

            self.train_epoch(idx_permutation)
            self.sess.run(self.model.increment_cur_epoch_tensor)

        # saving final model
        print('Saving final model...')
        self.model.global_step_tensor = int(987654321)
        self.model.save(self.sess)
        print('Done!')


    def train_epoch(self, idx_permutation):
        """
        Runs operations associated with one training epoch
        """
        loop = range(self.num_iter_per_epoch - 1)

        # MI estimation at the begin of every epoch
        for k in range(self.config.num_iter_MI):
            self.accumulated_MI_step()

        # train net
        for j in loop:
            idx = idx_permutation[j * self.config.batch_size:(j + 1) * self.config.batch_size]
            self.train_step(idx)

            # keep MI estimation up-to-date
            for q in range(self.config.num_iter_MI):
                self.accumulated_MI_step()


            # evaluate model performance
            if j % self.check == 0:

                mi_loss = self.evaluate_mi()
                ts_acc, ts_loss = self.model.evaluate_model(self.sess, self.data, split='test', n_chunks=10)
                eb1_acc, eb1_loss = self.model.evaluate_model(self.sess, self.imdb_eb1, split='train', n_chunks=10)
                eb2_acc, eb2_loss = self.model.evaluate_model(self.sess, self.imdb_eb2, split='train', n_chunks=10)

                summaries_dict = {
                    'MI': mi_loss,
                    'eb1_acc': eb1_acc,
                    'eb1_loss': eb1_loss,
                    'eb2_acc': eb2_acc,
                    'eb2_loss': eb2_loss,
                    'ts_acc': ts_acc
                }

                self.step += self.check

                print('Epoch: {}, Step: {}, MI_loss:{:10.4f}, eb1_acc: {:10.4f}, eb2_acc: {:10.4f}, ts_acc: {:10.4f}, eb1_loss: {:10.4f}, eb2_loss: {:10.4f}'.format(
                        self.cur_epoch, self.step, mi_loss, eb1_acc, eb1_loss, eb2_acc, eb2_loss, ts_acc))
                self.logger.summarize(self.step, summaries_dict=summaries_dict)


    def train_step(self, idx_batch):
        """
        Runs operations associated with a single training step (iteration)
        """
        batch_x, batch_y, batch_c = next(self.data.next_batch(idx_batch))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.c: batch_c, self.model.is_training: True}

        self.sess.run([self.model.C_train_op, self.model.update_ops], feed_dict=feed_dict)
        self.sess.run([self.model.M_train_op_c, self.model.update_ops], feed_dict=feed_dict)


    def evaluate_mi(self, N_chunks=10):
        """
        Evaluates the Mutual Information between features and bias
        """
        mi_ = 0
        for i in range(N_chunks):
            idx_batch = range(6000)[i * self.config.batch_size:(i + 1) * self.config.batch_size]
            batch_x, batch_y, batch_c = next(self.data.next_batch(idx_batch))

            feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.c: batch_c,
                         self.model.is_training: False}

            mi_ += self.sess.run(self.model.M_loss, feed_dict=feed_dict)
        return - mi_ / N_chunks


    def accumulated_MI_step(self):
        for i in range(self.config.num_iter_accumulation):
            idx_batch = np.random.choice(self.index_list, self.config.batch_size)
            batch_x, batch_y, batch_c = next(self.data.next_batch(idx_batch))

            feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.c: batch_c,
                         self.model.is_training: True}
            dummy = self.sess.run(self.model.accum_ops, feed_dict=feed_dict)

        self.sess.run(self.model.M_train_op_m)
        self.sess.run(self.model.zero_ops)
