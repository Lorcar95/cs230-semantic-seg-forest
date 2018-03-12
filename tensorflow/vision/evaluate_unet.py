"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_forest import input_forest
from model.model_unet import model_unet
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger


ext = '.npy'
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments_unet/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/split_FOREST',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "test_forest")

    # Get the filenames from the test set
    test_filenames = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir)
                        if f.endswith(ext)]
    test_images = [f for f in test_filenames if f.endswith('_image'+ext)]
    test_labels = [f for f in test_filenames if f.endswith('_label'+ext)]
    test_images.sort()
    test_labels.sort()

    # specify the size of the evaluation set
    params.eval_size = len(test_images)

    # create the iterator over the dataset
    test_inputs = input_forest(False, test_images, test_labels, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_unet('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
