"""Split the SIGNS dataset into train/dev/test and resize images to 64x64.

The SIGNS dataset comes in the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and dev sets.
Because we don't have a lot of images and we want that the statistics on the dev set be as
representative as possible, we'll take 20% of "train_signs" as dev set.
"""

import argparse
import random
import os

from shutil import copyfile
from PIL import Image
from tqdm import tqdm

IMAGE_PATH = '_split_rasters'
LABEL_PATH = '_forest_rasters'
IMAGE_PREFIX = 'img'
IMAGE_SUFFIX = '.TIF'
LABEL_PREFIX = 'mask'
LABEL_SUFFIX = '0.TIF'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/FOREST', help="Directory with the FOREST dataset")
parser.add_argument('--output_dir', default='data/split_FOREST', help="Where to write the new data")


def process_and_save(filename, output_dir):
    label_file = filename
    image_file = filename.replace(LABEL_PATH, IMAGE_PATH) \
                        .replace(LABEL_PREFIX, IMAGE_PREFIX) \
                        .replace(LABEL_SUFFIX, IMAGE_SUFFIX)

    """Process the image contained in `filename` and save it to the `output_dir`"""
    # label = Image.open(label_file)
    # image = Image.open(image_file)

    # Use bilinear interpolation instead of the default "nearest neighbor" method
    # image = image.resize((size, size), Image.BILINEAR)

    # Save images to output directory
    # label.save(os.path.join(output_dir, label_out))
    # image.save(os.path.join(output_dir, image_out))

    label_out = label_file.split('/')[-1].replace(LABEL_PREFIX, '').replace(LABEL_SUFFIX, '_label.tif')
    label_out = os.path.join(output_dir, label_out)
    image_out = image_file.split('/')[-1].replace(IMAGE_PREFIX, '').replace(IMAGE_SUFFIX, '_image.tif')
    image_out = os.path.join(output_dir, image_out)
    copyfile(label_file, label_out)
    copyfile(image_file, image_out)

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Get the filenames in the labels directory
    label_dir = os.path.join(args.data_dir, LABEL_PATH)
    filenames = os.listdir(label_dir)
    filenames = [os.path.join(label_dir, f) for f in filenames if f.endswith(LABEL_SUFFIX)]

    # Split the images in 'train_signs' into 80% train, 10% dev, 10% test
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    train_split = int(0.8 * len(filenames))
    dev_split = int(0.9*len(filenames))
    train_filenames = filenames[:train_split]
    dev_filenames = filenames[train_split:dev_split]
    test_filenames = filenames[dev_split:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_forest'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            process_and_save(filename, output_dir_split)

    print("Done building dataset")
