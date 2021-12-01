
import numpy as np
import nibabel as nib
import os
import tensorflow as tf
import pickle
import shutil
from matplotlib import pyplot as plt
import segmentation_models_3D as sm
from natsort import natsorted

files_path = os.path.join(os.getcwd(), "drive", "MyDrive", "Colab Notebooks", "thesis")
import sys
sys.path.append(files_path)

import preprocess as pp


# filename format is modelType_epochs_batch_lr_dropout_args.h5
def save_model_hist(model_name:str, save_model_dir:str, train_model, train_model_hist):
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    saved_model_folder_path = os.path.join(save_model_dir, model_name)
    if not os.path.exists(saved_model_folder_path):
        os.mkdir(saved_model_folder_path)
    train_model.save(os.path.join(saved_model_folder_path, (model_name + '.h5')))
    with open(os.path.join(saved_model_folder_path, (model_name + '_history')), 'wb') as fp:
        pickle.dump(train_model_hist.history, fp)
    print("model and model history saved")


# load model
def model_load(model_type:str, model_name:str, save_model_dir:str):
    saved_model_folder_path = os.path.join(save_model_dir, model_name)
    if model_type == "unet_w_bbone":
        model_loaded = tf.keras.models.load_model(os.path.join(saved_model_folder_path, (model_name + '.h5')), 
                                                               custom_objects={'dice_loss': sm.losses.DiceLoss(), 
                                                                               'focal_loss':sm.losses.CategoricalFocalLoss(), 
                                                                               'iou_score': sm.metrics.IOUScore(),
                                                                               'f1-score': sm.metrics.FScore()})
    elif model_type == "unet_wo_bbone":
        model_loaded = tf.keras.models.load_model(s.path.join(saved_model_folder_path, (model_name + '.h5')))
    
    print("model successfully loaded")
    

# function to plot the required and relevant plots for both types of Unet models
def model_plots(model_history, fig_size=(10, 10)):
    fig_dict = {6: (2, 3), 4: (2, 2)}
    keys = np.asarray(list(model_history.history.keys()))
    keys = keys.reshape(fig_dict[keys.size])
    fig, ax = plt.subplots(fig_dict[keys.size][0], fig_dict[keys.size][0], figsize=fig_size)
    for row in range(0, fig_dict[keys.size][1]):
        for col in range(0, fig_dict[keys.size][1]):
            ax[row, col].plot(np.arange(1, (len(model_history.history[keys[0, 0]]) + 1), 1), model_history.history[keys[row, col]])
            ax[row, col].set_title(keys[row, col])
            ax[row, col].set_xlabel("epochs")

# get label values
def get_labels(true_masks, pred_masks):
    for i in range(len(true_masks)):
        print(f"""
    SERIAL NUMBER {i + 1}:
        true labels: {pp.nominal_labels(np.unique(true_masks[i]))}
        predicted labels: {pp.nominal_labels(np.unique(pred_masks[i]))}\n
    """)

# revert to 3d from categorical
def revert_categorical(pred_cat): 
    return np.argmax(pred_cat, axis=-1)

# convert prediction to nifti
def convert_to_nifti(preds_3d:np.array, model_name:str, save_model_dir:str):
    saved_model_folder_path = os.path.join(save_model_dir, model_name)
    for i in range(len(preds_3d)):
        img = nib.Nifti1Image(preds_3d[i], np.eye(4))
        # img = nib.Nifti1Image(preds_3d, ct_vol.affine)
        nib.save(img, os.path.join(saved_model_folder_path, 'predicted_mask_' + str(i)))
    print("all masks saved as nifti (3D)")

# pipeline
def model_predictions(model_name:str, model, test_X, test_Y_cat, save_model_dir:str, make_nifti:bool=False):
    test_Y_3d = np.array([revert_categorical(i) for i in test_Y_cat])
    preds_cat = model.predict(test_X)
    preds_3d = np.array([revert_categorical(i) for i in preds_cat])
    print(f"for label reference: {pp.color_dict}")
    print(f"""
    SHAPES:
        test_Y_3d: {test_Y_3d.shape},
        preds_cat: {preds_cat.shape},
        preds_3d: {preds_3d.shape},
    """)
    iou_val = tf.metrics.MeanIoU(num_classes=5)
    iou_val.update_state(test_Y_cat, preds_cat)
    print(f"""
    predicted_mean_iou_score: {iou_val.result().numpy()}
    """)
    if make_nifti:
        convert_to_nifti(preds_3d, model_name, save_model_dir)
    get_labels(test_Y_3d, preds_3d)
    return test_Y_3d, preds_cat, preds_3d

# Plotting functions
# plot GD and predictions
def plot_truth_pred_cat(test_X, test_Y_cat, preds_cat, class_ind:int, sample_ind:int, slice_start:int,  slice_end:int, fig_size:tuple=(6, 6)):
        for i in range(slice_start, slice_end):
            fig, ax = plt.subplots(1, 3, figsize=fig_size)

            ax[0].imshow(test_X[sample_ind][:, :, i])
            ax[0].set_title('ct_volume')

            ax[1].imshow(test_Y_cat[sample_ind][:, :, i, class_ind])
            ax[1].set_title('ground truth')

            ax[2].imshow(preds_cat[sample_ind][:, :, i, class_ind])
            ax[2].set_title('predicted')

            plt.tight_layout()

def plot_truth_pred_3d(test_X, test_Y_3d, preds_3d, sample_ind:int, slice_start:int, slice_end:int):
        for i in range(slice_start, slice_end):
            fig, ax = plt.subplots(1, 3, figsize=(6, 6))

            ax[0].imshow(test_X[sample_ind][:, :, i])
            ax[0].set_title('ct_volume')

            ax[1].imshow(test_Y_3d[sample_ind][:, :, i])
            ax[1].set_title('ground truth')

            ax[2].imshow(preds_3d[sample_ind][:, :, i])
            ax[2].set_title('predicted')

            plt.tight_layout()


