import tensorflow as tf
import tensorflow.contrib.slim as slim


class Digit_model():
    def __init__(self, config):
        self.config = config
        self.build_model()
        self.init_saver()

        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()


    def build_model(self):
        """
        Defines tf graph (models + losses)
        """
        # parameters
        lr = self.config.lr_C
        lmb = self.config.lmb
        dim_c = 3 * self.config.dim_c  # 3 color channels * resolution single channel

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
        self.c = tf.placeholder(tf.float32, [None, dim_c], name='c')
        self.is_training = tf.placeholder(tf.bool, shape=())

        logits, z = self.build_net(self.x, self.is_training)

        c_bar = tf.random_shuffle(self.c)

        joint = tf.concat([self.c, z], axis=1)
        margn = tf.concat([c_bar, z], axis=1)

        t = self.M(joint)
        et = tf.exp(self.M(margn, reuse=True))

        # MI estimation loss
        self.M_loss = -(tf.reduce_mean(t) - tf.log(tf.reduce_mean(et)))

        # task loss
        self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))

        # evaluate model
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # optimizer
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        tvars = tf.trainable_variables()

        c_vars = [var for var in tvars if 'Net' in var.name]
        m_vars = [var for var in tvars if 'Mine' in var.name]

        # MINE optimizer
        self.M_train_op_m = tf.train.AdamOptimizer(lr).minimize(self.M_loss, var_list=m_vars)

        # optimizer of min[ loss_ce + lmb*loss_mi ]
        M_optimizer = tf.train.AdamOptimizer(lr * lmb)
        C_optimizer = tf.train.AdamOptimizer(lr)

        # gradient clipping
        ggu = tf.gradients(self.C_loss, c_vars)
        ggm = tf.gradients(-self.M_loss, c_vars)

        for i, (gu, gm) in enumerate(zip(ggu, ggm)):
            if gm == None: continue
            gu_ = tf.norm(gu)
            gm_ = tf.norm(gm)
            g_ = tf.minimum(gu_, gm_)

            # gradients of mi_loss are normalized
            ggm[i] = tf.multiply(g_, tf.divide(gm, gm_))

        # train operations
        ga_and_vars = list(zip(ggm, c_vars))
        self.M_train_op_c = M_optimizer.apply_gradients(grads_and_vars=ga_and_vars)

        ggu_and_vars = list(zip(ggu, c_vars))
        self.C_train_op = C_optimizer.apply_gradients(grads_and_vars=ggu_and_vars)


    def M(self, inp, h_dim=64, reuse=False):
        """
        Defines Statistics Network (MINE)
        """
        with tf.variable_scope('Mine', reuse=reuse):
            fc1 = slim.fully_connected(inputs=inp, num_outputs=h_dim, activation_fn=tf.nn.leaky_relu)
            fc2 = slim.fully_connected(inputs=fc1, num_outputs=h_dim, activation_fn=tf.nn.leaky_relu)
            fc3 = slim.fully_connected(inputs=fc2, num_outputs=h_dim, activation_fn=tf.nn.leaky_relu)
            out = slim.fully_connected(inputs=fc3, num_outputs=1, activation_fn=None)

            return out


    def build_net(self, x, is_training):
        """
        Defines network architecture (feature extractor + classifier)
        """
        with tf.variable_scope('Net', reuse=False):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=tf.nn.relu,
                                    padding='VALID'):
                    net = slim.conv2d(x, 64, 5, scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], padding='SAME')

                    net = slim.conv2d(net, 64, 5, scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], padding='SAME')

                    net = tf.contrib.layers.flatten(net)
                    net = slim.fully_connected(net, 512, scope='fc1')

                    net = slim.fully_connected(net, self.config.dim_z, scope='fc2')

                    # feature to unbias
                    z = net

                    # logit layer
                    logits = slim.fully_connected(z, 10, activation_fn=None, scope='logits')

        return logits, z


    def evaluate_model(self, sess, X, y):
        """
        Evalautes the model on given datapoints.
        """
        batch_size = 1000
        acc = 0
        for i in range(10):
            batch_x = X[i * batch_size: (i + 1) * batch_size] / 255.
            batch_y = y[i * batch_size: (i + 1) * batch_size]

            feed_dict = {self.x: batch_x,
                         self.y: batch_y}

            acc += sess.run(self.accuracy, feed_dict)

        return acc / 10


    def init_saver(self):
        """
        Initializes the tensorflow saver that will be used in saving the checkpoints.
        """
        self.saver = tf.train.Saver(max_to_keep=5)


    def save(self, sess):
        """
        Saves the checkpoint in the path defined in the config file.
        """
        print("Saving model...")
        print(self.config.checkpoint_dir, self.global_step_tensor)
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved.")


    def load(self, sess):
        """
        Loads latest checkpoint from the experiment path defined in the config file.
        """
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded.")


    def init_cur_epoch(self):
        """
        Initializes a tensorflow variable to use it as epoch counter.
        """
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)


    def init_global_step(self):
        """
        Initializes a tensorflow variable to use it as global step counter.
        Do not forget to add the global step tensor to the tensorflow trainer.
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')



