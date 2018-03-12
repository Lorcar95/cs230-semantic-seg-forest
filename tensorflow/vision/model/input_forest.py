"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import numpy as np

from skimage import io


def _decode_function(image, label):
    """Obtain the image and label from the filenames (for both training and validation).

    The following operations are applied:
        - Decode the image from tiff format
        - Convert to float and to range [0, 1]
    """
    image_decoded = np.load(image.decode())
    image_decoded = image_decoded.astype(np.float32)

    # TODO: let this accomodate class > 2
    label_decoded = np.load(label.decode())
    label_decoded[label_decoded > 1] = 0
    label_class = np.zeros((label_decoded.shape[0], label_decoded.shape[1], 2), dtype=np.int64)
    label_class[...,1] = label_decoded
    label_class[...,0] = 1-label_decoded

    return image_decoded, label_class

def _parse_function(image, label, size, channels, classes):
    """Clean up the image.

    The following operations are applied:
        - Resize to size by size by channels
        - Convert to float and to range [0, 1]
    """
    image_reshape = tf.reshape(image, [size, size, channels])
    label_reshape = tf.reshape(label, [size, size, classes])

    image_norm = tf.clip_by_value(tf.divide(image_reshape, 10000), 0, 1)

    return image_norm, label_reshape


def train_preprocess(image, label, use_random_flip):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    # if use_random_flip:
    #     image = tf.image.random_flip_left_right(image)

    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # # Make sure the image is still in [0, 1]
    # image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def input_forest(is_training, images, labels, params):
    """Input function for the FOREST dataset.

    The image filenames have format "{id}_image.tif"
    The label filenames have format "{id}_label.tif"
    For instance: "data_dir/27_image.tif" and "data_dir/27_label.tif".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        images: (list) filenames of the images, as ["{id}_image.tif"...]
        labels: (list) corresponding filenames of labels, as ["{id}_label.tif"...]
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(images)
    assert len(images) == len(labels), "images and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    decode_fn = lambda f, l: tuple(tf.py_func(_decode_function, [f, l], [tf.float32, tf.int64]))
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size, params.image_channels, params.image_classes)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(images), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(decode_fn, num_parallel_calls=params.num_parallel_calls)
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(images), tf.constant(labels)))
            .map(decode_fn)
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
