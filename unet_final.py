# Main part for training the model for performing semantic segmentation of the medical images (ct-scans)

# imports (standard)
# for segmentation models to be imported successfully, keras applications and keras_preprocessing need to be installed
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
import segmentation_models_3D as sm

# folder paths for the training, validation and test dataset
code_folder = os.getcwd()
x_path = os.path.join(code_folder, 'data_X')
y_path = os.path.join(code_folder, 'data_Y')
paths = (x_path, y_path)
saved_models_path = os.path.join(code_folder,'saved_models')

# import the preprocess, unet3d and postprocess based on the folders above
import preprocess as pp
import unet3d as un3
import postprocess as po
import model_train as mt

def main():
    """
    main function which initiates the deep learning pipeline with several options for constants, parameters and hyperparameters
    
    """
    # only if different values are required for these parameters (set to default as mentioned)
    pp.resize_depth_shape = (144, 102, 320)
    pp.req_height_width = (128, 128)
    pp.win_wid_val = 1000
    pp.win_level_val = 400

    # load the X_data (ct-scans) and Y_data (masks)
    X_data, Y_data = pp.get_all_data(paths, preprocess=True, expand_dim=True, verbose=False, norm_option='mean_norm')

    # sanity check for the pre-processed volumes and masks
    plot_data = False
    if plot_data:
        pp.plot_all_data(X_data, Y_data, sample_start=7, sample_end=9, slices_start=140, slices_end=145, class_ind=1)

    # splitting of the data
    # training - testing split
    train_test_ratio = 0.95
    train_X, train_Y, test_X, test_Y = pp.generate_data(X_data, Y_data, train_test_ratio)

    # to recover the memory (delete original arrays)
    del X_data, Y_data

    # training - validation split
    train_val_ratio = 0.9
    train_X, train_Y, val_X, val_Y = pp.generate_data(train_X, train_Y, train_val_ratio)

    # sanity check for shape and number of samples in each type of dataset
    print(f"""
    TRAINING SET: 
        X: {train_X.shape}
        Y: {train_Y.shape} 
        
    VALIDATION SET:
        X: {val_X.shape} 
        Y: {val_Y.shape} 

    TEST SET:
        X: {test_X.shape} 
        Y: {test_Y.shape}
    """)

    # define common parameters for the model
    no_classes = 5
    no_epochs = 50
    b_size = 1
    learn_rate = 2e-4
    dropout_val = 0.2
    img_dim = (train_X[0].shape[0], train_X[0].shape[1], train_X[0].shape[2], train_X[0].shape[3])
    model_params_str = str(no_epochs) + '_' + str(b_size) + '_' + "{:.0e}".format(learn_rate) + '_' + str(dropout_val)


    # start model training/learning
    trained_model, trained_model_hist = mt.model_train((train_X, train_Y, val_X, val_Y),
                                                        model_type="unet_w_bbone", 
                                                        num_classes=no_classes, 
                                                        lr=learn_rate, 
                                                        batch=b_size, 
                                                        num_epochs=no_epochs, 
                                                        dropout_value=dropout_val, 
                                                        img_shape=img_dim,
                                                        shuffle_bool=True
                                                        )

    # save the model and the history
    po.save_model_hist(model_params_str, saved_models_path, trained_model, trained_model_hist)

    # plot the losses and metrics
    po.model_plots(trained_model_hist, fig_size=(10, 10))

    # predictions
    test_Y_3d_vals, preds_cat_vals, preds_3d_vals = po.model_predictions(model_params_str, trained_model, test_X, test_Y, saved_models_path, make_nifti=True)

    # plot categorical (per class)
    po.plot_truth_pred_cat(test_X,
                        test_Y,
                        preds_cat_vals,
                        class_ind=2, 
                        sample_ind=2,
                        slice_start=180, 
                        slice_end=185
                        )

    # plot 3d (not categorical)
    po.plot_truth_pred_3d(test_X,
                        test_Y_3d_vals,
                        preds_3d_vals,
                        sample_ind=0,
                        slice_start=115, 
                        slice_end=120
                        )

    # load the saved model
    load_model = False
    if load_model:
        loaded_model = po.model_load("unet_w_bbone", model_params_str, saved_models_path)

if __name__ == '__main__':
    main()

######################################################################################################

# extras
# img_width = train_X[0].shape[0]
# img_height = train_X[0].shape[1]
# img_depth = train_X[0].shape[2]
# no_channels = 3

# tensorboard
# import datetime
# logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# %reload_ext tensorboard
# %tensorboard --logdir logs

# dice_loss = sm.losses.DiceLoss(class_weights=np.array()
# categorical_focal_loss = sm.losses.CategoricalFocalLoss()
# categorical_focal_dice_loss = categorical_focal_loss + dice_loss


# loading model with custom loss functions
# load model
# model_loaded = tf.keras.models.load_model(os.path.join(data_path, 'vgg19_50_2_0001.h5'), 
#                                    custom_objects={'dice_loss': sm.losses.DiceLoss(), 
#                                                    'focal_loss':sm.losses.CategoricalFocalLoss(), 
#                                                    'iou_score': sm.metrics.IOUScore(),
#                                                    'f1-score': sm.metrics.FScore()})