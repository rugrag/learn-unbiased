import tensorflow as tf
import numpy.random as npr
import os

from imdb_loader import Imdb_loader
from imdb_model import Imdb_model
from imdb_trainer import Imdb_trainer
from imdb_parser import get_config

from logger import Logger


def main():
    
    tf.reset_default_graph()

    dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir, 'data')

    config = get_config(exp_dir=os.path.join(dir, 'experiments'))

    # set seed based on run ID
    seed = 213*config.run 
    npr.seed(seed)

    # create tensorflow session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    # create your data generator
    digit_data = Imdb_loader(data_dir, config)

    print('data loaded')

    # create an instance of the model you want
    model = Imdb_model(config)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = Imdb_trainer(sess, model, digit_data, config, logger, data_dir)

    # load model if exists
    #model.load(sess)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()

