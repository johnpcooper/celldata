from skimage.filters import rank, threshold_otsu, sobel_v
from skimage.morphology import disk, remove_small_objects
from scipy import ndimage
import time
from skimage import transform
from skimage.morphology import skeletonize
import numpy as np
import math
import logging
import os

log = logging.getLogger(__name__)

class Constants(object):
    FIFTEEN_DEGREES_IN_RADIANS = 0.262
    ACCEPTABLE_SKEW_THRESHOLD = 5.0
    NUM_CATCH_CHANNELS = 28
    
def create_vertical_segments(image_data):
    """
    Creates a binary image with blobs surrounding areas that have a lot of vertical edges
    """
    # find edges that have a strong vertical direction
    vertical_edges = sobel_v(image_data)
    # Sepearate out the areas where there is a large amount of vertically
    # oriented stuff
    return _segment_edge_areas(vertical_edges)
    
def _segment_edge_areas(edges, disk_size=9, mean_threshold=200, min_object_size=500):
    """
    
    Takes a greyscale image (with brighter colors corresponding to edges) and returns
    a binary image with high edge density and black indicates low density
    
    param image_data: a 2D numpy array
    
    """
    
    # convert the greyscale edge information into black and white image
    threshold = threshold_otsu(edges)
    # Filter out the edge data below the threshold, effectively removing some noise
    raw_channel_areas = edges <= threshold
    # smooth out the data
    channel_areas = rank.mean(raw_channel_areas, disk(disk_size)) < mean_threshold
    # remove specks and blobs that are the result of artifacts
    clean_channel_areas = remove_small_objects(channel_areas, min_size=min_object_size)
    # Fill in any areas that re completely surrounded by the areas (hopefully) covering
    # the channels
    return ndimage.binary_fill_holes(clean_channel_areas)


# from fylm/service/rotation.py
def _determine_rotation_offset(image):
    """
    Finds rotational skew so that the sides of the central trench are (nearly) perfectly vertical.
    :param image:   raw image data in a 2D (i.e. grayscale) numpy array
    :type image:    np.array()
    """
    segmentation = create_vertical_segments(image)
    # Draw a line that follows the center of the segments at each point, which should be roughly vertical
    # We should expect this to give us four approximately-vertical lines, possibly with many gaps in each line
    skeletons = skeletonize(segmentation)
    # Use the Hough transform to get the closest lines that approximate those four lines
    hough = transform.hough_line(skeletons, np.arange(-Constants.FIFTEEN_DEGREES_IN_RADIANS,
                                                      Constants.FIFTEEN_DEGREES_IN_RADIANS,
                                                      0.0001))
    # Create a list of the angles (in radians) of all of the lines the Hough transform produced, with 0.0 being
    # completely vertical
    # These angles correspond to the angles of the four sides of the channels, which we need to correct for
    angles = [angle for _, angle, dist in zip(*transform.hough_line_peaks(*hough))]
    if not angles:
        log.warn("Image skew could not be calculated. The image is probably invalid.")
        return 0.0
    else:
        # Get the average angle and convert it to degrees
        offset = sum(angles) / len(angles) * 180.0 / math.pi
        if offset > Constants.ACCEPTABLE_SKEW_THRESHOLD:
            log.warn("Image is heavily skewed. Check that the images are valid.")
        return offset
    
def rotate_image(image, offset):
    """
    
    Return an image (np.array()) rotated by the number of degrees
    returned by _determine_rotation_offset(image)
    
    """
    
    return transform.rotate(image, offset)

import logging
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from skimage.transform import AffineTransform

# from fylm/service/registration.py
def _determine_registration_offset(base_image, uncorrected_image):
    """
    
    Finds the translational offset required to align this image with all others in the stack.
    Returns dx, dy adjustments in pixels *but does not change the image!*
    
    :param base_image:   a 2D numpy array that the other image should be aligned to
    :param uncorrected_image:   a 2D numpy array
    :returns:   float, float
    
    """

    # Get the dimensions of the images that we're aligning
    base_height, base_width = base_image.shape
    uncorrected_height, uncorrected_width = uncorrected_image.shape

    # We take the area that roughly corresponds to the catch channels. This has two benefits: one, it
    # speeds up the registration significantly (as it scales linearly with image size), and two, if
    # a large amount of debris/yeast/bacteria/whatever shows up in the central trench, the registration
    # algorithm goes bonkers if it's considering that portion of the image.
    # Thus we separately find the registration for the left side and right side, and average them.
    left_base_section = base_image[:, int(base_width * 0.1): int(base_width * 0.3)]
    left_uncorrected = uncorrected_image[:, int(uncorrected_width * 0.1): int(uncorrected_width * 0.3)]
    right_base_section = base_image[:, int(base_width * 0.7): int(base_width * 0.9)]
    right_uncorrected = uncorrected_image[:, int(uncorrected_width * 0.7): int(uncorrected_width * 0.9)]

    # 
    left_dy, left_dx = register_translation(left_base_section, left_uncorrected, upsample_factor=20)[0]
    right_dy, right_dx = register_translation(right_base_section, right_uncorrected, upsample_factor=20)[0]

    return (left_dy + right_dy) / 2.0, (left_dx + right_dx) / 2.0

def translate_image(uncorrected_image, translational_offset):
    x = translational_offset[1]
    y = translational_offset[0]
    new_image = transform.warp(uncorrected_image, transform.AffineTransform(translation=(-x, -y)))
    return new_image

import tifffile as tf
import tkinter as tk
import tkinter.filedialog as dia
import skimage.io as io
from skimage.util import img_as_uint

def align_images(fov_path):

    fov_slice_filenames = os.listdir(fov_path)

    images = []
    for filename in fov_slice_filenames:
        image = tf.imread(fov_path + '/%s' % filename)
        images.append(image)

    # create a list of offsets using _determine_rotation_offset()
    rotational_offsets = []
    # create an index variable to print so the user can see progress
    index = 0
    for image in images:
        index = index + 1
        print("Determining rotation offset %d of %d" % (index, len(images)))
        vis_channel = image[0]
        rotational_offsets.append(_determine_rotation_offset(vis_channel))

    # create a list of rotationally aligned images rotated according to the rotational_offsets list
    rotated_vis_images = []
    rotated_yfp_images = []
    rotated_dsred_images = []

    index = 0 # again, progress bar index
    for i in range(0, len(images)):

        vis_channel = images[i][0]
        yfp_channel = images[i][1]
        dsred_channel = images[i][2]

        index = index + 1
        print("Rotating image %d of %d" % (index, len(images)))
        # if rotational offset was not determined for this image, it will have been set to 0.
        # if that's the case, use the rotational offset from 10 images ago.
        if rotational_offsets[i] != 0.0:

	        rotated_vis_images.append(rotate_image(vis_channel, rotational_offsets[i]))
	        rotated_yfp_images.append(rotate_image(yfp_channel, rotational_offsets[i]))
	        rotated_dsred_images.append(rotate_image(dsred_channel, rotational_offsets[i]))

        elif rotational_offsets[i] == 0.0:
        	print("Rotational alignment failed for image %s, using rotation offset from image %s" % (i, i-10))
	        rotated_vis_images.append(rotate_image(vis_channel, rotational_offsets[i-10]))
	        rotated_yfp_images.append(rotate_image(yfp_channel, rotational_offsets[i-10]))
	        rotated_dsred_images.append(rotate_image(dsred_channel, rotational_offsets[i-10]))

    # create list of translational offsets and define a progress bar index
    index = 0
    translational_offsets = []
    for image in rotated_vis_images:
        index  = index + 1
        print("Determining registration offset %d of %d" % (index, len(images)))
        translational_offsets.append(_determine_registration_offset(rotated_vis_images[0], image))

    # create a list of translationally aligned images translated according to the translational_offsets list
    translated_vis_images = []
    translated_yfp_images = []
    translated_dsred_images = []

    index = 0
    for i in range(0, len(translational_offsets)):
        index = index + 1
        print("Translating image %d of %d" % (index, len(images)))

        # check whether the translational offset that was determined above is going to shift the image out of frame
        # if it does cause a shift out of frame, just use the previous translational offset
    	x_offset = translational_offsets[i][1]
    	y_offset = translational_offsets[i][0]

    	offset_magnitude = np.sqrt(np.power(x_offset, 2) + np.power(y_offset, 2))

    	# this would be buggy if there are multiple bad alignments in a row and I use i-1. So arbitrarily I'm 
    	# saying go 10 images back. It would be useful to add scanning functionality so I can go back
    	# and find the last sub 20 pixel offset_magnitude
        if offset_magnitude <= 20:

	        translated_vis_images.append(translate_image(rotated_vis_images[i], translational_offsets[i]))
	        translated_yfp_images.append(translate_image(rotated_yfp_images[i], translational_offsets[i]))
	        translated_dsred_images.append(translate_image(rotated_dsred_images[i], translational_offsets[i]))

        elif offset_magnitude >= 20:
        	print("Registration for image %s failed, using translational_offset from image %s" % (i, i-10))
	        translated_vis_images.append(translate_image(rotated_vis_images[i], translational_offsets[i-10]))
	        translated_yfp_images.append(translate_image(rotated_yfp_images[i], translational_offsets[i-10]))
	        translated_dsred_images.append(translate_image(rotated_dsred_images[i], translational_offsets[i-10]))

    return (translated_vis_images, translated_yfp_images, translated_dsred_images)

def save_stacks(vis, yfp, dsred, save_path, fov_name):
    
    vis_con = io.concatenate_images(img_as_uint(vis))
    tf.imsave(save_path + '/%s_bf_stack.tif' % fov_name, vis_con)
    print("BF stack saved")

    yfp_con = io.concatenate_images(img_as_uint(yfp))
    tf.imsave(save_path + '/%s_yfp_stack.tif' % fov_name, yfp_con)
    print("yfp stack saved")

    dsred_con = io.concatenate_images(img_as_uint(dsred))
    tf.imsave(save_path + '/%s_dsred_stack.tif' % fov_name, dsred_con)
    print("dsred stack saved")