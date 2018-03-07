"""Train the model"""

import argparse
import logging
import os
import random

import tensorflow as tf

from model.input_forest import input_forest
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_unet import model_unet
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments_unet/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/split_FOREST',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, 'train_forest')
    dev_data_dir = os.path.join(data_dir, 'dev_forest')
    test_data_dir = os.path.join(data_dir, 'test_forest')

    # Get the filenames from the train and dev sets
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
                       if f.endswith('.tif')]
    eval_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)
                      if f.endswith('.tif')]

    # Get list of images and corresponding labels
    train_images = [f for f in train_filenames if f.endswith('_image.tif')]
    train_labels = [f for f in train_filenames if f.endswith('_label.tif')]
    train_images.sort()
    train_labels.sort()
    eval_images = [f for f in eval_filenames if f.endswith('_image.tif')]
    eval_labels = [f for f in eval_filenames if f.endswith('_label.tif')]
    eval_images.sort()
    eval_labels.sort()

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_images)
    params.eval_size = len(eval_images)

    # Create the two iterators over the two datasets
    train_inputs = input_forest(True, train_images, train_labels, params)
    eval_inputs = input_forest(False, eval_images, eval_labels, params)

    # Define the model
    logging.info("Creating the model...")

    # Sanity check the dataset model
    # with tf.Session() as sess:
    #     sess.run(train_inputs['iterator_init_op'])
    #     num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
    #     print(num_steps)
    #     for i in range(num_steps):
    #         image, label = sess.run([train_inputs['images'], train_inputs['labels']])
    #         print("Iter: "+ str(i))
    #         print(image.shape)
    #         print(label.shape)
    #         print(image[0])
    #         print(label[0])

    train_model_spec = model_unet('train', train_inputs, params)
    eval_model_spec = model_unet('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
