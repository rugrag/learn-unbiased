import numpy.random as npr
import tensorflow as tf


class Digit_trainer():
    def __init__(self, sess, model, data, config, logger, data_dir):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.data_dir = data_dir
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.num_iter_per_epoch = self.data.N_samples // self.config.batch_size

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.cur_epoch = cur_epoch
            idx_permutation = npr.permutation(range(self.data.N_samples))

            self.train_epoch(idx_permutation)
            self.sess.run(self.model.increment_cur_epoch_tensor)

        # save model at the end of training
        self.model.save(self.sess)


    def train_epoch(self, idx_permutation):

        loop = range(self.num_iter_per_epoch -1 )


        # MINE training at the beginning of every epoch
        for i in loop:
            idx = idx_permutation[i * self.config.batch_size:(i + 1) * self.config.batch_size]
            ce_loss, mi_loss, acc = self.train_step(idx, ce=False, mine=True)

        # train net
        for j in loop:
            idx = idx_permutation[j * self.config.batch_size:(j + 1) * self.config.batch_size]
            ce_loss, mi_loss, acc = self.train_step(idx, ce=True, mine=False)

            # train mine
            for q in range(self.config.num_iter_MI):
                q = q%(self.num_iter_per_epoch -1)
                idx = idx_permutation[q * self.config.batch_size:(q + 1) * self.config.batch_size]
                ce_loss, mi_loss, acc = self.train_step(idx, ce=False, mine=True)


        # evaluate model performances
        if self.model.cur_epoch_tensor.eval(self.sess) % 1 == 0:

            tr_acc = self.model.evaluate_model(self.sess, self.data.x_train, self.data.y_train)
            ts_acc = self.model.evaluate_model(self.sess, self.data.x_test, self.data.y_test)

            cur_it = self.model.cur_epoch_tensor.eval(self.sess)
            summaries_dict = {
                'ce_loss': ce_loss,
                'MI': mi_loss,
                'tr_acc': tr_acc,
                'ts_acc': ts_acc
            }

            print('Epoch: {}, CE_loss: {:10.4f}, MI_loss:{:10.4f}, tr_acc: {:10.4f}, ts_acc: {:10.4f}'.format(
                    self.cur_epoch, ce_loss, mi_loss, tr_acc, ts_acc))
            self.logger.summarize(self.cur_epoch, summaries_dict=summaries_dict)





    def train_step(self, idx_batch, ce=True, mine=True):
        batch_x, batch_y, batch_c = next(self.data.next_batch(idx_batch))
        batch_x = batch_x / 255.

        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.c: batch_c,
                     self.model.is_training: True}

        if ce:
            self.sess.run([self.model.C_train_op, self.model.update_ops], feed_dict=feed_dict)
            self.sess.run([self.model.M_train_op_c, self.model.update_ops], feed_dict=feed_dict)

            ce_loss, mi_loss, acc = self.sess.run(
                [self.model.C_loss, self.model.M_loss, self.model.accuracy],
                feed_dict=feed_dict)

        if mine:
            _, ce_loss, mi_loss, acc = self.sess.run(
                [self.model.M_train_op_m, self.model.C_loss, self.model.M_loss, self.model.accuracy],
                feed_dict=feed_dict)

        return ce_loss, -mi_loss, acc
