#
# Copyright (c) 2017 Mellowain
# This software is released under the MIT license. See the attached LICENSE file for details.
# Written by Milind Deore <tomdeore@gmail.com>
#
# This code snippet use imgaug(https://github.com/aleju/imgaug) library to augment the given input image
# and convert them in the same directory structure as input was.
#

import os
import sys
import argparse
from datetime import datetime
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import scipy.misc
import glob
import shutil

# Debug will enable GUI
MW_DEBUG = False



ia.seed(1)


#
# Create output directory for all augumented images.
#
def create_dir(dir_name):
	if not os.path.exists(dir_name):
    		os.makedirs(dir_name)


#
# add a random value from the range (-30, 30) to all the channels of
# input images (e.g. to the R, G and B channels)
#
seq_rand_val_RGB_chann = iaa.Sequential(
    [
  	iaa.WithChannels(
	    channels=[0, 1, 2],
  	    children=iaa.Add((-90, 90))),
    ],
)

def mw_rand_val_RGB_chan(N_dim, images_collection, class_outdir):
    print '[' + sys._getframe().f_code.co_name + ']'
    for i in range(0, len(images_collection)):
	img = np.array([images_collection[i] for _ in range(N_dim)], dtype=np.uint8)
        single_img = np.stack(img, axis = 0)
	images_aug = seq_rand_val_RGB_chann.augment_images(single_img)
	for i in range(N_dim):
	    if MW_DEBUG == True:
	        plt.imshow(images_aug[i])
	        plt.show()
	    aug_fn = class_outdir + '/' + datetime.utcnow().strftime("%Y-%m-%d_%H%M%S.%f") + '.jpg'
	    scipy.misc.imsave(aug_fn, images_aug[i])
	    if i > N_dim:
		break


#
# Add a value of -10 to 10 to each pixel.
#
seq_rand_val_each_pixel = iaa.Sequential(
    [
        iaa.Add((-50, 50), per_channel=0.5),
    ],
)

def mw_rand_val_each_pixel(N_dim, images_collection, class_outdir):
    print '[' + sys._getframe().f_code.co_name + ']'
    for i in range(0, len(images_collection)):
	img = np.array([images_collection[i] for _ in range(N_dim)], dtype=np.uint8)
        single_img = np.stack(img, axis = 0)
	images_aug = seq_rand_val_each_pixel.augment_images(single_img)
	for i in range(N_dim):
	    if MW_DEBUG == True:
	        plt.imshow(images_aug[i])
	        plt.show()
	    aug_fn = class_outdir + '/' + datetime.utcnow().strftime("%Y-%m-%d_%H%M%S.%f") + '.jpg'
	    scipy.misc.imsave(aug_fn, images_aug[i])
	    if i > N_dim:
		break

#
# Change brightness of images (50-150% of original value).
#
seq_brightness_per_img = iaa.Sequential(
    [
        iaa.Multiply((0.5, 2.5), per_channel=0.5),
    ],
)

def mw_brightness_per_img(N_dim, images_collection, fn):
    print '[' + sys._getframe().f_code.co_name + ']'
    for i in range(0, len(images_collection)):
	img = np.array([images_collection[i] for _ in range(N_dim)], dtype=np.uint8)
        single_img = np.stack(img, axis = 0)
	images_aug = seq_brightness_per_img.augment_images(single_img)
	for i in range(N_dim):
	    if MW_DEBUG == True:
	        plt.imshow(images_aug[i])
	        plt.show()
	    aug_fn = class_outdir + '/' + datetime.utcnow().strftime("%Y-%m-%d_%H%M%S.%f") + '.jpg'
	    scipy.misc.imsave(aug_fn, images_aug[i])
	    if i > N_dim:
		break


#
# Improve or worsen the contrast of images.
#
seq_contrast_per_img = iaa.Sequential(
    [
        iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
    ],
)

def mw_seq_contrast_per_img(N_dim, images_collection, class_outdir):
    print '[' + sys._getframe().f_code.co_name + ']'
    for i in range(0, len(images_collection)):
	img = np.array([images_collection[i] for _ in range(N_dim)], dtype=np.uint8)
        single_img = np.stack(img, axis = 0)
	images_aug = seq_contrast_per_img.augment_images(single_img)
	for i in range(N_dim):
	    if MW_DEBUG == True:
	        plt.imshow(images_aug[i])
	        plt.show()
	    aug_fn = class_outdir + '/' + datetime.utcnow().strftime("%Y-%m-%d_%H%M%S.%f") + '.jpg'
	    scipy.misc.imsave(aug_fn, images_aug[i])
	    if i > N_dim:
		break


#
# Flip images vertically.
#
seq_flipud = iaa.Sequential(
    [
        iaa.Flipud(1), # vertically flip 100% of all images
    ],
)

def mw_flipud(N_dim, images_collection, class_outdir):
    print '[' + sys._getframe().f_code.co_name + ']'
    for i in range(0, len(images_collection)):
	img = np.array([images_collection[i] for _ in range(N_dim)], dtype=np.uint8)
        single_img = np.stack(img, axis = 0)
	images_aug = seq_flipud.augment_images(single_img)
	for i in range(N_dim):
	    if MW_DEBUG == True:
	        plt.imshow(images_aug[i])
	        plt.show()
	    aug_fn = class_outdir + '/' + datetime.utcnow().strftime("%Y-%m-%d_%H%M%S.%f") + '.jpg'
	    scipy.misc.imsave(aug_fn, images_aug[i])
	    if i > N_dim:
		break

#
# Flip images horizontally.
#
seq_fliplr = iaa.Sequential(
    [
        iaa.Fliplr(1), # horizontally flip 100% of all images
    ],
)

def mw_fliplr(N_dim, images_collection, class_outdir):
    print '[' + sys._getframe().f_code.co_name + ']'
    for i in range(0, len(images_collection)):
	img = np.array([images_collection[i] for _ in range(N_dim)], dtype=np.uint8)
        single_img = np.stack(img, axis = 0)
	images_aug = seq_fliplr.augment_images(single_img)
	for i in range(N_dim):
	    if MW_DEBUG == True:
	        plt.imshow(images_aug[i])
	        plt.show()
	    aug_fn = class_outdir + '/' + datetime.utcnow().strftime("%Y-%m-%d_%H%M%S.%f") + '.jpg'
	    scipy.misc.imsave(aug_fn, images_aug[i])
	    if i > N_dim:
		break


#
# Crop and pad 25% of image
#
seq_crop_and_pad = iaa.Sequential(
    [
        iaa.CropAndPad(percent=(-0.25, 0.25))
    ],
)

def mw_crop_and_pad(N_dim, images_collection, class_outdir):
    print '[' + sys._getframe().f_code.co_name + ']'
    for i in range(0, len(images_collection)):
	img = np.array([images_collection[i] for _ in range(N_dim)], dtype=np.uint8)
        single_img = np.stack(img, axis = 0)
	images_aug = seq_crop_and_pad.augment_images(single_img)
	for i in range(N_dim):
	    if MW_DEBUG == True:
	        plt.imshow(images_aug[i])
	        plt.show()
	    aug_fn = class_outdir + '/' + datetime.utcnow().strftime("%Y-%m-%d_%H%M%S.%f") + '.jpg'
	    scipy.misc.imsave(aug_fn, images_aug[i])
	    if i > N_dim:
		break


#
# Apply affine transformations to some of the images
# - scale to 80-120% of image height/width (each axis independently)
# - translate by -20 to +20 relative to height/width (per axis)
# - rotate by -45 to +45 degrees
# - shear by -16 to +16 degrees
#
seq_affine_transformations = iaa.Sequential(
    [
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16)
        ),
    ],
)

def mw_affine_transformations(N_dim, images_collection, class_outdir):
    print '[' + sys._getframe().f_code.co_name + ']'
    for i in range(0, len(images_collection)):
	img = np.array([images_collection[i] for _ in range(N_dim)], dtype=np.uint8)
        single_img = np.stack(img, axis = 0)
	images_aug = seq_affine_transformations.augment_images(single_img)
	for i in range(N_dim):
	    if MW_DEBUG == True:
	        plt.imshow(images_aug[i])
	        plt.show()
	    aug_fn = class_outdir + '/' + datetime.utcnow().strftime("%Y-%m-%d_%H%M%S.%f") + '.jpg'
	    scipy.misc.imsave(aug_fn, images_aug[i])
	    if i > N_dim:
		break


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
        print "Copy done for subdir: " + dst

#
# main processing starts ...
#

parser = argparse.ArgumentParser(description=' -- Mellowain image augmentation tool -- ')
parser.add_argument('directory', help='input root directory on which augmentation has to run')
args = parser.parse_args()

rootdir = args.directory
print ' Root directory : ' + rootdir


#
# Processed images are stored here.
#
output_directory = "./mw_output/"
create_dir(output_directory)


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # Go deep until we find files, that the directory we are interested in.
        fn = os.path.join(subdir, file)
	cwd = os.path.basename(os.path.dirname(fn))
	print ' Class name: ' + cwd
	#
	# collect all the files from INPUT folder
	#
	images_collection = io.ImageCollection(subdir + '/*.JPG')
	#
	# create class OUTPUT folder.
	#
	class_outdir = output_directory + cwd
	create_dir(class_outdir)
        copytree(subdir, class_outdir)

	##
	####  Perform image augmentation ####
	##
	mw_rand_val_RGB_chan(10, images_collection, class_outdir)
	# Add a value of -50 to 50 to each pixel
        ##mw_rand_val_each_pixel(2, images_collection, class_outdir)
	# Change brightness of images (50-150% of original value)
	mw_brightness_per_img(10, images_collection, class_outdir)
        # Improve or worsen the contrast of images.
	mw_seq_contrast_per_img(10, images_collection, class_outdir)
        # Crop and pad the images
	mw_crop_and_pad(10, images_collection, class_outdir)
	# affine transformation on the images
	mw_affine_transformations(5, images_collection, class_outdir)

        break

print "Augementation Done!"
