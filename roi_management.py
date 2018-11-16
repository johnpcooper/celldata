import tifffile as tf
import tkinter as tk
import tkinter.filedialog as dia
import skimage as ski
import skimage.io as io
import matplotlib as plt
import numpy as np
import pandas as pd

# utilities (get filepath and get the image to be analyzed)

def set_fp(prompt):
    
    """ Return the path to a file of interest. Call with the prompt you would like to 
    display in the dialog box."""
    
    # create the dialog box and set the fn
    root = tk.Tk()
    fp = dia.askopenfilename(parent=root, title=prompt)
    root.destroy() # very important to destroy the root object, otherwise python 
    # just keeps running in a dialog box
    
    return fp # return the path to the file you just selected

def set_image_stack(prompt):
    
    """ Return a tf.imread() object referring to the image stack"""
    
    fp = set_fp(prompt)
    print("Path to the stack .tif selected:\n%s" % fp)
    image_stack = tf.imread(fp)
    # let the user see the tif stack they just instantiated
    print("Displaying 0th image in this stack:")
    #io.imshow(image_stack[0])
    
    return image_stack

# instantiate nad modify dataframe objects for the master cell index 
# and create a dataframe for each cell in the master index

def set_master_cells_df(prompt):

    """Return a dataframe read from the master cells index .csv"""
    
    # define the path to the index .csv
    master_cells_fp = set_fp(prompt)
    # define the filename for the master expt index
    master_cells_df = pd.read_csv(master_cells_fp)
    
    return master_cells_df

def set_cells_dfs(master_cells_df):
    
    """Return a list of sub-dataframes of the master_cells_df. The sub-dataframes
       contain data for only one cell according to its sub_coord"""
    
    # create the cells_dfs list
    cells_dfs = []
    
    # add to the cells_dfs a dataframe at index i for every
    # unique value in the master_cells_df['sub_coord']
    for i in master_cells_df['sub_coord'].unique():
        # set the logic mask to a sub-dataframe of master_cells_df containing
        # only values whose 'sub_coord' value is value
        print("value is ", i)
        logic_mask = (master_cells_df['sub_coord'] == i)
        cells_dfs.append(master_cells_df[logic_mask])
        print("cells_dfs is now %d elements long" % len(cells_dfs))
        
    return cells_dfs

# Functions to iterate through the ROIS of one of the cells defined above
# with ROI parameters defined in the master index
def set_cell_crop_params(image_stack, cells_dfs, cell_sub_coord):
    
    """ Return and save to the cwd a cropped image of the cell corresponding to
        cell_sub_coord by making a sub-matrix of the image_stack """
    
    # set the upper bounds as Y_ub and X_ub. 'Y' and 'X' are currently the lower bounds    
    cells_dfs[cell_sub_coord]['Y_ub'] = cells_dfs[cell_sub_coord]['Y'] + cells_dfs[cell_sub_coord]['Height']
    cells_dfs[cell_sub_coord]['X_ub'] = cells_dfs[cell_sub_coord]['X'] + cells_dfs[cell_sub_coord]['Width']

def set_cropped_cell_stack(image_stack, cells_dfs, cell_sub_coord):
    
    """ Return a cropped stack made of the ROIs in cells_dfs """
    
    y_lb = cells_dfs[cell_sub_coord]['Y']
    y_ub = cells_dfs[cell_sub_coord]['Y_ub']
    x_lb = cells_dfs[cell_sub_coord]['X']
    x_ub = cells_dfs[cell_sub_coord]['X_ub']
    
    cropped_image_stack = []
    
    # iterate over ROI number
    for i in range(cells_dfs[cell_sub_coord]['Pos'].index.min(), cells_dfs[cell_sub_coord]['Pos'].index.max() + 1):
        
        print("Cropping slices with ROI %d of cell %d: " % (i, cell_sub_coord))
        
        if cells_dfs[cell_sub_coord]['Pos'].index.max() > i >= cells_dfs[cell_sub_coord]['Pos'].index.min():
                          
            first_stack_index = cells_dfs[cell_sub_coord]['Pos'][i] - 1
            last_stack_index = cells_dfs[cell_sub_coord]['Pos'][i + 1] - 1

            # if true, then this is a single position ROI
            if (first_stack_index - last_stack_index) == 1:
                
                stack_index = first_stack_index
                
                print("Cropping stack index %d (single index ROI)" % stack_index)
                
                source_image = image_stack[stack_index]
                cropped_image = source_image[y_lb[i]: y_ub[i], x_lb[i]: x_ub[i]]
                cropped_image_stack.append(cropped_image)                
                
            elif (first_stack_index - last_stack_index) != 1:                
                # iterate over the slices of image_stack between the first two positions
                for stack_index in range(first_stack_index, last_stack_index):
                    # this loop stops just short of last_stack_index, which will be the first index
                    # of the next ROI loop
                    print("Cropping stack index %d (multi-index ROI)" % stack_index)
                                       
                    source_image = image_stack[stack_index]
                    cropped_image = source_image[y_lb[i]: y_ub[i], x_lb[i]: x_ub[i]]                
                    cropped_image_stack.append(cropped_image)
            
            else:
                print("Error: Stack index %d may contain multiple ROIs" % first_stack_index)
        
        # add a cropped stack index at the last position of the image_stack. This may
        # potentially cause problems if the final stack index is part of a mult-index ROI
        elif i == cells_dfs[cell_sub_coord]['Pos'].index.max():
            first_stack_index = cells_dfs[cell_sub_coord]['Pos'][i] - 1
            
            stack_index = first_stack_index
            
            print("Cropping stack index %d (final stack index)" % stack_index)
            source_image = image_stack[stack_index]
            cropped_image = source_image[y_lb[i]: y_ub[i], x_lb[i]: x_ub[i]]                
            cropped_image_stack.append(cropped_image)
            
        else: 
            pass
                
    return cropped_image_stack

# The images must be the same shape so that they can be converted back into
# a stack using skimage.io.concatenate_images(cropped_image_stack). I accomplish
# this by adding background value pixels to expand images until they are the 
# same size as the largest image in the cropped_image_stack

def set_slice_shapes_array(cropped_stack):
    
    """ Return a np.array() containing the shape of each slice in
        cropped_stack. For example, slice_shapes_array[:, 0] is a 
        list of the heights of each slice. """
    
    cropped_stack = cropped_stack.copy()
    slice_shapes_list = []
    
    for i in range(0, len(cropped_stack)):
        # Create a list of tuples, one tuple for each slice, where
        # the first element of the tuple is the height (# rows)
        # and the second element is width (# columns) of that slice

        slice_shapes_list.append(cropped_stack[i].shape)
    # convert the list of tuples into a 2d np.array() object
    slice_shapes_array = np.array(slice_shapes_list, dtype='uint16')
    
    return slice_shapes_array

def set_max_dims(slice_shapes_array):
    
    """ Return a tuple which is (max height in slice_shapes_array, max width in slice_shapes_array)"""
    
    # maximum value in the heights column
    height_max = slice_shapes_array[:, 0].max()
    # maximum value in the widths column
    width_max = slice_shapes_array[:, 1].max()
    
    print("Largest slice in stack is %d rows by %d columns" % (height_max, width_max))
    
    return (height_max, width_max)

def set_slice_offsets(cropped_stack, slice_shapes_array, height_max, width_max):
    
    """ Return a tuple which is (list of height_max - slice_shapes_array[:, 0],
        list of width_max - slice_shapes_array[:, 1]) """
    
    # define the differences in height and width between each slice and the maxima
    
    slice_height_offsets = []
    slice_width_offsets = []
    
    for i in range(0, len(cropped_stack)):

        slice_height_offsets.append(height_max - slice_shapes_array[i, 0])
        slice_width_offsets.append(width_max - slice_shapes_array[i, 1])
        
    return (slice_height_offsets, slice_width_offsets)


def set_resized_stack(cropped_stack, slice_height_offsets, slice_width_offsets):
    
    """ Return the cropped_stack with added zeros to make each slice the same 
        size as the largest slice in the stack """
    
    resized_stack = []
    
    for i in range(0, len(cropped_stack)):
        
        image = cropped_stack[i]
        resized_image = image.copy()
        image_h_offset = slice_height_offsets[i]
        image_w_offset = slice_width_offsets[i]
        
        # important that dtype = 'unit16' because default dtype is 64 bit float for 
        # np.full. Imagej can't read 64 bit images, so I'm just keeping the images 
        # at 16 bit integers because that's the original bit depth at which they
        # are acquired
        h_filler_value = resized_image[-1, :].mean()
        h_filler = np.full((image_h_offset, resized_image.shape[1]), h_filler_value, dtype='uint16')
        resized_image = np.append(resized_image, h_filler, axis=0)
        
        w_filler_value = resized_image[:, -1].mean()
        w_filler = np.full((resized_image.shape[0], image_w_offset), w_filler_value, dtype='uint16')
        resized_image = np.append(resized_image, w_filler, axis=1)
        
        resized_stack.append(resized_image)
        
    return resized_stack

# define functions to first run the cropping image process and then to 
# resized them

def run(cell_index):
    
    """ Return a cropped stack according to the cell index parameters found in master index .csv """
   
    master_cells_df = set_master_cells_df("Choose the .csv for the master index of this experiment")
    image_stack = set_image_stack("Choose the dsred .tif stack ")
    
    cells_dfs = set_cells_dfs(master_cells_df)
    set_cell_crop_params(image_stack, cells_dfs, cell_index)
    cropped_stack = set_cropped_cell_stack(image_stack, cells_dfs, cell_index)
        
    return cropped_stack

def run_resize(cropped_stack, image_title):

    slice_shapes_array = set_slice_shapes_array(cropped_stack)
    height_max, width_max = set_max_dims(slice_shapes_array)

    slice_height_offsets, slice_width_offsets = set_slice_offsets(cropped_stack, slice_shapes_array, height_max, width_max)

    resized_stack = set_resized_stack(cropped_stack, slice_height_offsets, slice_width_offsets)
    final_resized_stack = io.concatenate_images(resized_stack)

    # save the stack in the current working directory with the title passed above
    io.imsave("%s.tif" % image_title, final_resized_stack)
    
    return final_resized_stack