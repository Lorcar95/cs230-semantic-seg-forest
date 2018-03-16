"""Split the FOREST dataset into train/dev/test and resize images to 256x256.

The FOREST dataset comes in the following format:
    _forest_rasters/
        mask00.TIF
        ...
    _split_rasters/
        img0.TIF
        ...

Original images have size (256, 256).
"""

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

from shutil import copyfile
from tqdm import tqdm
from skimage import io
from PIL import Image

IMAGE_PATH = '_split_rasters/'
LABEL_PATH = '_forest_rasters/'
IMAGE_PREFIX = 'img'
IMAGE_SUFFIX = '.TIF'
LABEL_PREFIX = 'mask'
LABEL_SUFFIX = '0.TIF'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/LANDSAT_FULL', help="Directory with the FOREST dataset")
parser.add_argument('--output_dir', default='data/LANDSAT_FULL_aug', help="Where to write the new data")


def process_and_save(filename, output_dir, augment=False):
    images = {}

    image_file = filename
    image_num = filename.replace(IMAGE_PATH, '') \
                        .replace(IMAGE_PREFIX, '') \
                        .replace(IMAGE_SUFFIX, '') \
                        .split('/')[-1]
    label_num = str(int(image_num)+1)
    label_file = image_file.replace(image_num, label_num) \
                        .replace(IMAGE_PATH, LABEL_PATH) \
                        .replace(IMAGE_PREFIX, LABEL_PREFIX) \
                        .replace(IMAGE_SUFFIX, LABEL_SUFFIX)

    """Process the image contained in `filename` and save it to the `output_dir`"""
    label = io.imread(label_file)
    label = label-1
    image = io.imread(image_file)
    images['_orig'] = (image, label)

    if augment:
        # Flip image
        images['_flip'] = (np.flip(image,0), np.flip(label,0))

        # Rotate images
        new_items = []
        for tag, ims in images.items():
            im, lbl = ims
            for i in range(3):
                new_items.append((tag+"_rot"+str(i+1), (np.rot90(im,i+1), np.rot90(lbl,i+1))))

        for item in new_items:
            images[item[0]] = item[1]

        # Add Gaussian noise to images
        new_items = []
        for tag, ims in images.items():
            im, lbl = ims
            for i in range(1):
                noise = np.random.normal(0,6.5,image.shape)
                new_items.append((tag+"_gaus"+str(i+1), (im+noise, lbl)))

        for item in new_items:
            images[item[0]] = item[1]

    # # Plot augmented images
    # n_per_plot = 4
    # fig, ax = plt.subplots(n_per_plot,2,sharex=True,sharey=True,figsize=(12,5))
    # i = 0
    # for tag, ims in images.items():
    #     idx = i%n_per_plot
    #     img, lbl = ims
    #     img_rgb = np.flip(img[:,:,:3], 2)
    #     ax[idx,0].imshow(img_rgb/256)
    #     ax[idx,1].imshow(lbl)
    #     ax[idx,0].set_title(tag+" Image")
    #     ax[idx,1].set_title(tag+" Label")
    #     i+=1
    #     if (i%n_per_plot == 0):
    #         fig, ax = plt.subplots(n_per_plot,2,sharex=True,sharey=True,figsize=(12,5))
    # plt.show()

    # Create base output file names
    label_out = str(int(label_file.split('/')[-1].replace(LABEL_PREFIX, '').replace(LABEL_SUFFIX, ''))-1)
    label_out = os.path.join(output_dir, label_out)
    image_out = image_file.split('/')[-1].replace(IMAGE_PREFIX, '').replace(IMAGE_SUFFIX, '')
    image_out = os.path.join(output_dir, image_out)

    # Save images to output directory
    for tag, ims in images.items():
        im, lbl = ims
        if lbl.shape[0] != 256 or lbl.shape[1] != 256:
            print(im.shape)
            print(image_out+tag)
            print(lbl.shape)
            break
        np.save(image_out+tag+"_image.npy", im)
        np.save(label_out+tag+"_label.npy", lbl)


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Get the filenames in the labels directory
    image_dir = os.path.join(args.data_dir, IMAGE_PATH)
    filenames = os.listdir(image_dir)
    filenames = [os.path.join(image_dir, f) for f in filenames if f.endswith(IMAGE_SUFFIX)]

    # Split the images in 'train_signs' into 80% train, 10% dev, 10% test
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    train_split = int(0.8 * len(filenames))
    dev_split = int(0.9 * len(filenames))
    train_filenames = filenames[:train_split]
    dev_filenames = filenames[train_split:dev_split]
    test_filenames = filenames[dev_split:]

    print(len(filenames))
    print(len(train_filenames))
    print(len(dev_filenames))
    print(len(test_filenames))

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
            if split == 'train':
                process_and_save(filename, output_dir_split, augment=True)
            else:
                process_and_save(filename, output_dir_split)

    print("Done building dataset")
