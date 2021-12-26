# post-processing applied to the ct-scans and the masks prior to the model training

# imports
import numpy as np
import nibabel as nib
import os
import tensorflow as tf
import pickle
import shutil
from matplotlib import pyplot as plt
import segmentation_models_3D as sm
from natsort import natsorted
import preprocess as pp

# filename format is modelType_epochs_batch_lr_dropout_args.h5
def save_model_hist(model_name:str, save_model_dir:str, train_model, train_model_hist):
    """
    saves the model and model history at the given location

    Inputs:
        model_name: the name of the model in the format as specified above
        save_model_dir: directory path where the model has to be saved
        train_model: the trained model that needs to be saved
        train_model_hist: the model history corresponding to the trained model

    Outputs:
        saves the model and its history at the specified location and prints a statement on successful saving
    """
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
    """
    loads the model and model history from the given location

    Inputs:
        model_type: model with backbone (for encoder) (needs custom loss and metrics to be loaded) or not (wiht scratch)
        model_name: the name of the model in the format as specified above
        save_model_dir: directory path where the model has to be saved
    
    Outputs:
        loads the model and its history from the specified location and prints a statement on successful loading
    """
    saved_model_folder_path = os.path.join(save_model_dir, model_name)
    if model_type == "unet_w_bbone":
        model_loaded = tf.keras.models.load_model(os.path.join(saved_model_folder_path, (model_name + '.h5')), 
                                                               custom_objects={'dice_loss': sm.losses.DiceLoss(), 
                                                                               'focal_loss':sm.losses.CategoricalFocalLoss(), 
                                                                               'iou_score': sm.metrics.IOUScore(),
                                                                               'f1-score': sm.metrics.FScore()})
    elif model_type == "unet_wo_bbone":
        model_loaded = tf.keras.models.load_model(os.path.join(saved_model_folder_path, (model_name + '.h5')))
    
    print("model successfully loaded")
    

# function to plot the required and relevant plots for both types of Unet models
def model_plots(model_history, fig_size=(10, 10)):
    """
    plots the loss function and the metrics for the trained model

    Inputs:
        model_history: the history of the trained model
        fig_size: the size of the plots

    Outputs:
        plots the required data of the trained model
    """

    fig_dict = {6: (2, 3), 4: (2, 2)}
    keys = np.asarray(list(model_history.history.keys()))
    keys = keys.reshape(fig_dict[keys.size])
    fig, ax = plt.subplots(fig_dict[keys.size][0], fig_dict[keys.size][1], figsize=fig_size)
    for row in range(0, fig_dict[keys.size][0]):
        for col in range(0, fig_dict[keys.size][1]):
            ax[row, col].plot(np.arange(1, (len(model_history.history[keys[0, 0]]) + 1), 1), model_history.history[keys[row, col]])
            ax[row, col].set_title(keys[row, col])
            ax[row, col].set_xlabel("epochs")
            ax[row, col].set_ylim(bottom=0)

# get label values
def get_labels(true_masks, pred_masks):
    """
    Fetches the labels of the original and the predicted masks for comparison
    
    Inputs:
        true_masks: the original Y data
        pred_masks: the predicted masks

    Outputs:
        prints the true and predicted labels in the given corresponding masks
    """
    for i in range(len(true_masks)):
        print(f"""
    SERIAL NUMBER {i + 1}:
        true labels: {pp.nominal_labels(np.unique(true_masks[i]))}
        predicted labels: {pp.nominal_labels(np.unique(pred_masks[i]))}\n
    """)

# revert to 3d from categorical
def revert_categorical(pred_cat): 
    """
    converts back from categorical

    Inputs:
        pred_cat: the predictions as given by the trained model
    
    Outputs:
        returns the predicted mask with the maximum value in the last axis (collapses the last dimension - reverts back from categorical)
    """
    return np.argmax(pred_cat, axis=-1)

# convert prediction to nifti
def convert_to_nifti(preds_3d:np.array, model_name:str, save_model_dir:str):
    """
    generates the nifti file for the predictions (predicted masks) in the given directory

    Inputs:
        preds_3d: the 3d dimensional predicted mask
        model_name: the name of the model for which the mask needs to be converted to the nifti file
        save_model_dir: the directory where the nifti file needs to be saved

    Outputs:
        generates the nifti file and saves it to the given directory and prints the succesful generation and saving message
    """
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    saved_model_folder_path = os.path.join(save_model_dir, model_name)
    if not os.path.exists(saved_model_folder_path):
        os.mkdir(saved_model_folder_path)
    for i in range(len(preds_3d)):
        img = nib.Nifti1Image(preds_3d[i], np.eye(4))
        # img = nib.Nifti1Image(preds_3d, ct_vol.affine)
        nib.save(img, os.path.join(saved_model_folder_path, 'predicted_mask_' + str(i)))
    print("all masks saved as nifti (3D)")

# pipeline
def model_predictions(model_name:str, model, test_X, test_Y_cat, save_model_dir:str, make_nifti:bool=False):
    """
    makes the predictions for the test data using the trained model

    Inputs:
        model_name: the name of the model 
        model: the model using which the predictions would be made
        test_X: ct_scans in the test dataset (X)
        test_Y: masks in the test dataset (Y)
        save_model_dir: directory where the model is saved
        make_nifti: enable or disable nifti file generation for the predicted masks
    
    Outputs:
        returns the test masks, predicted masks in categorical form and the predicted masks in the 3D form and generates and saves nifti if enabled
    """
        
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
    """
    plots the volume (ct-scan), mask (ground truth) and the corresponding masks in the categorical form

    Inputs:
        test_X: ct-scan in the test dataset (X)
        test_Y_cat: mask in the test datasetin the categorical form(Y)
        preds_cat: predictions in the categorical form 
        class_ind: required class index
        sample_ind: required sample index
        slice_start: start value/index of the slice for plotting
        slice_end: end value/index of the slice for plotting
        fig_size: size of the plots

    Outputs:
        plots the required ct-scan and its mask from the test data and corresponding prediction

    """
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
    """
    plots the volume (ct-scan), mask (ground truth) and the corresponding masks in the 3D form

    Inputs:
        test_X: ct-scan in the test dataset (X)
        test_Y_3D: mask in the test dataset in the 3D form (Y)
        preds_3D: predictions in the 3D form 
        sample_ind: required sample index
        slice_start: start value/index of the slice for plotting
        slice_end: end value/index of the slice for plotting

    Outputs:
        plots the required ct-scan and its mask from the test data and corresponding prediction
        
    """
    for i in range(slice_start, slice_end):
        fig, ax = plt.subplots(1, 3, figsize=(6, 6))

        ax[0].imshow(test_X[sample_ind][:, :, i])
        ax[0].set_title('ct_volume')

        ax[1].imshow(test_Y_3d[sample_ind][:, :, i])
        ax[1].set_title('ground truth')

        ax[2].imshow(preds_3d[sample_ind][:, :, i])
        ax[2].set_title('predicted')

        plt.tight_layout()
