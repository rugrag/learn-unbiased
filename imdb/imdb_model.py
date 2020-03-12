import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1


class Imdb_model():

    def __init__(self, config):
        
        self.config = config
        self.build_model()
        self.init_saver()

        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    def build_model(self):

        # parameters
        lr_C = self.config.lr_C
        lr_M = self.config.lr_M
        lmb = self.config.lmb

        # placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, 2], name='y')
        self.c = tf.placeholder(tf.float32, [None, 12], name='c')
        self.is_training = tf.placeholder(tf.bool, shape=())

        # Build graph
        self.logits, self.z = self.build_net(self.x, self.is_training)
        self.z = tf.squeeze(self.z, [1, 2])

        d_bar = tf.random_shuffle(self.d)

        joint = tf.concat([self.d, self.z], axis=1)
        margn = tf.concat([d_bar, self.z], axis=1)


        t = self.M(joint)
        et = tf.exp(self.M(margn, reuse=True))

        # Loss
        self.M_loss = -(tf.reduce_mean(t) - tf.log(tf.reduce_mean(et)))

        self.C_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=tf.squeeze(self.logits)))

        # Evaluate model
        self.preds = tf.argmax(tf.squeeze(self.logits), 1)  # predictions
        self.gts = tf.argmax(self.y, 1)  # ground truth values
        correct_pred = tf.equal(self.preds, self.gts)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Optimizer
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        t_vars = tf.trainable_variables()

        c_vars = [var for var in t_vars if 'resnet_v1_50' in var.name]
        m_vars = [var for var in t_vars if 'Mine' in var.name]


        # gradient accumulation
        mine_opt = tf.train.GradientDescentOptimizer(lr_M)

        cc = tf.constant(1 / self.config.num_iter_accumulation)

        # Creation of a list of variables with the same shape as the trainable ones initialized with 0s
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in m_vars]

        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        # Calls the compute_gradients function of the optimizer to obtain... the list of gradients
        gvs = mine_opt.compute_gradients(self.M_loss, m_vars)

        # Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
        self.accum_ops = [accum_vars[i].assign_add(tf.scalar_mul(cc, gv[0])) for i, gv in enumerate(gvs)]

        # Define the training step (part with variable value update)
        self.M_train_op_m = mine_opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])



        # optimizer di min[ loss_ce + lmb*loss_mi ]
        M_optimizer = tf.train.AdamOptimizer(lr_C * lmb)
        C_optimizer = tf.train.AdamOptimizer(lr_C)

        # gradient clipping
        ggu = tf.gradients(self.C_loss,  c_vars)  # some gradients are None ('resnet_v1_50/logits/weights:0' AND 'resnet_v1_50/logits/biases:0')
        ggm = tf.gradients(-self.M_loss, c_vars)  # BISOGNA METTERE IL - O NO??????????

        for i, (gu, gm) in enumerate(zip(ggu, ggm)):
            # clip

            if gm is None:
                continue

            gu_ = tf.norm(gu)
            gm_ = tf.norm(gm)
            g_ = tf.minimum(gu_, gm_)

            # NORMALIZZA I GRADIENTI DI M_LOSS
            ggm[i] = tf.multiply(g_, tf.divide(gm, gm_))

        ga_and_vars = list(zip(ggm, c_vars))
        self.M_train_op_c = M_optimizer.apply_gradients(grads_and_vars=ga_and_vars)

        ggu_and_vars = list(zip(ggu, c_vars))
        self.C_train_op = C_optimizer.apply_gradients(grads_and_vars=ggu_and_vars)

    def M(self, inp, h_dim=512, reuse=False, name='Mine'):

        with tf.variable_scope(name, reuse=reuse):
            fc1 = slim.fully_connected(inputs=inp, num_outputs=h_dim, activation_fn=tf.nn.leaky_relu)
            fc2 = slim.fully_connected(inputs=fc1, num_outputs=h_dim, activation_fn=tf.nn.leaky_relu)
            fc3 = slim.fully_connected(inputs=fc2, num_outputs=h_dim, activation_fn=tf.nn.leaky_relu)
            out = slim.fully_connected(inputs=fc3, num_outputs=1, activation_fn=None)

            return out

        # senza BN

    def build_net(self, x, is_training):

        print('Building resnet-50')

        # network architecture
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            _, end_points = resnet_v1.resnet_v1_50(x, num_classes=2, is_training=is_training)

        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
            net = end_points['resnet_v1_50/block4']  # last bottleneck before logits

            with tf.variable_scope('resnet_v1_50'):
                z = slim.conv2d(net, self.config.dim_z, [7, 7], padding='VALID', activation_fn=tf.nn.relu,
                                         scope='bottleneck_layer')
                logits = slim.conv2d(z, 2, [1, 1], activation_fn=None, scope='logit_layer')

        return logits, z


    # TODO SISTEMARE
    def evaluate_model(self, sess, X, y):
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
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=5)

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        print(self.config.checkpoint_dir, self.global_step_tensor)
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')



