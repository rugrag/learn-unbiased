import tensorflow as tf
import numpy.random as npr
import os

from digit_loader import Digit_loader
from digit_model import Digit_model
from digit_trainer import Digit_trainer
from digit_parser import get_config

from logger import Logger


def main():
    tf.reset_default_graph()

    dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir, 'data')

    config = get_config(exp_dir=os.path.join(dir, 'experiments'))

    # set seed based on run ID
    seed = 213 * config.run
    npr.seed(seed)

    # create tensorflow session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    # create your data generator
    digit_data = Digit_loader(data_dir, config)

    print('data loaded')

    # create an instance of the model you want
    model = Digit_model(config)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = Digit_trainer(sess, model, digit_data, config, logger, data_dir)

    # load model if exists
    # model.load(sess)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()

