import numpy as np
import nibabel as nib
from natsort import natsorted
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

resize_depth_shape = (144, 102, 320)
req_height_width = (128, 128)

color_dict = {0: 'background',
              1: 'blue',
              2: 'green',
              3: 'red',
              4: 'yellow'}

def preprocess_ct(ct_scan:np.array, norm_option:int='standard_norm'):
    # ct_scan = resize_volume(ct_scan)
    
    ct_scan.resize(resize_depth_shape, refcheck=False)
    ct_scan = tf.image.resize(ct_scan, (req_height_width[0], req_height_width[1])).numpy()

    win_wid = 1000
    win_level = 400
    range_max = win_level + (win_wid / 2)
    range_min = win_level - (win_wid / 2)
    ct_scan[ct_scan <= range_min] = range_min
    ct_scan[ct_scan >= range_max] = range_max
    if norm_option == 'standard_norm':
        ct_scan = (ct_scan - range_min) / (range_max - range_min)
    elif norm_option == 'mean_norm':
        ct_scan = (ct_scan - np.mean(ct_scan)) / (range_max - range_min)
    elif norm_option == 'z_norm':
        ct_scan = (ct_scan - np.mean(ct_scan)) / (np.std(ct_scan))
    return ct_scan

def preprocess_ct_seg(ct_scan_seg:np.array):
    
    # resize the segmentation
    ct_scan_seg.resize(resize_depth_shape, refcheck=False) # commented out for above
    ct_scan_seg = tf.image.resize(ct_scan_seg, (req_height_width[0], req_height_width[1])).numpy().astype(np.uint64).astype(np.float32) # commented out for above
    
    return ct_scan_seg
    

# get the colors in segmentation
def nominal_labels(unique_vals:list):
    colors_present = []
    for i in unique_vals:
        colors_present.append(color_dict[int(i)])
    return colors_present


# get the relevant and most-used properties for a numpy array (all together)
def get_prop(x):
    if isinstance(x, np.ndarray):
        print(f"""
        shape: {x.shape}
        size: {x.size}
        num_dimension: {np.ndim(x)}
        dtype: {x.dtype}
        max_val: {np.max(x)}
        min_val: {np.min(x)}
        """)
    else:
        print(f"type: {type(x)}")


# expand dimensions of the ct scan
def expand_dim_ct(ct_vol):
    vol_exp_dim = np.stack((ct_vol,)*3, axis=-1)
    return vol_exp_dim


# expand the dimensions of the segmentation and convert to categorical
def expand_categ_seg(ct_seg):
    vol_1ch = np.expand_dims(ct_seg, axis=-1)
    vol_1ch_cat = tf.keras.utils.to_categorical(ct_seg, num_classes=5)
    return vol_1ch_cat

def get_all_data(paths:tuple, preprocess:bool=True, expand_dim:bool=True, verbose:bool=False, norm_option:int=1):
    x_path, y_path = paths
    X_data, Y_data = [], []
    folders = natsorted(os.listdir(x_path))
    actual_loaded = 0
    for serial_no, folder in zip(range(1, len(folders) + 1), folders):
        actual_loaded += 1
        ct_path = os.path.join(x_path, folder, folder + 'HRimage.nii.gz')
        seg_path = os.path.join(y_path, folder, folder + 'HRimage_seg.nii')

        # ct_scan load
        ct_vol = nib.load(ct_path)
        ct_vol_data = ct_vol.get_fdata()

        if preprocess:
            ct_vol_data = preprocess_ct(ct_vol_data, norm_option)
        if expand_dim:
            ct_vol_data = expand_dim_ct(ct_vol_data)
        
        X_data.append(ct_vol_data)
        
        # ct_scan mask load
        ct_seg = nib.load(seg_path)        
        ct_seg_data = ct_seg.get_fdata()
        ct_seg_data_unique = np.unique(ct_seg_data)

        if preprocess:
            ct_seg_data = preprocess_ct_seg(ct_seg_data)
        if expand_dim:
            ct_seg_data = expand_categ_seg(ct_seg_data)
        
        Y_data.append(ct_seg_data)

        if verbose:
            print(f"""
            SERIAL NUMBER: {serial_no}:
                Loaded: ct_volume {folder} | shape: {ct_vol_data.shape}
                Loaded: ct_scan_seg {folder} | shape: {ct_seg_data.shape} | unique_vals : {ct_seg_data_unique} | colors: {nominal_labels(ct_seg_data_unique.astype(np.uint8))}\n
            """)

            # print(f"Loaded: ct_volume {folder} | shape: {ct_vol_data.shape}")
            # print(f"Loaded: ct_scan_seg {folder} | shape: {ct_seg_data.shape} | unique_vals : {ct_seg_data_unique} | colors: {nominal_labels(ct_seg_data_unique.astype(np.uint8))}\n")

    print(f"{actual_loaded} Volumes & Masks Loaded!")
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    return X_data, Y_data

# split the data
def generate_data(X_data, Y_data, train_ratio:float):
    train_X, val_X, train_Y, val_Y = train_test_split(X_data, Y_data, train_size=train_ratio)
    return train_X, train_Y, val_X, val_Y


# for plotting from the original combined datasets
def plot_vols_masks(X_data:np.array, Y_data:np.array, vol_ind:int, slice_index:int):
    fig, ax = plt.subplots(1, 2, figsize=(4, 4))
    ax[0].imshow(X_data[vol_ind][:, :, slice_index])
    ax[0].set_title(f'{vol_ind + 1}: Volume')
    # plt.subplot(1, 3, 1)
    # plt.imshow(image)
    # plt.title('Volume')

    ax[1].imshow(Y_data[vol_ind][:, :, slice_index])
    ax[1].set_title(f'{vol_ind+ 1}: Mask')
    # plt.subplot(1, 3, 2)
    # plt.imshow(actual_gray)
    # plt.title('Mask')
    plt.tight_layout()

# to check the pre-processed volumes and masks - for this the expand_dim above should be set to false (3d and not categorical)
def plot_all_data(X_data:np.array, Y_data:np.array, sample_start:int, sample_end:int, slices_start:int, slices_end:int):
    for i in range(sample_start, sample_end):
        for j in range(slices_start, slices_end):
            plot_vols_masks(X_data, Y_data, i, j)
            plt.title(f"{i + 1} - {j + 1}")




