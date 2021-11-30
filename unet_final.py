# -*- coding: utf-8 -*-
"""unet_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x_MypbSWEAH6wW62NZHGFUD9CjWInd-1

### **Thesis Code, Author @Rajat Sharma** ###
### **Deep Learning Based Semeantic Segmentation of Bone Fragments in CT Scans**
"""

# imports
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
import segmentation_models_3D as sm

# folder paths for the training, validation and test dataset
code_folder = os.getcwd()
# data_path = os.path.join(code_folder, "crop_data")
x_path = os.path.join(code_folder, 'data_X')
y_path = os.path.join(code_folder, 'data_Y')
paths = (x_path, y_path)
saved_models_path = os.path.join(code_folder,'saved_models')
# sys.path.append(os.path.join(code_folder))
import preprocess as pp
import unet3d as un3
import postprocess as po

# only if different dimensions required
# these two parameters are set to default values as shown below

# pp.resize_depth_shape = (144, 102, 320) # original (mostly) 144, 102, 302
# pp.req_height_width = (128, 128)
X_data, Y_data = pp.get_all_data(paths, preprocess=True, expand_dim=True, verbose=False, norm_option='standard_norm')

# to check the pre-processed volumes and masks - for this the expand_dim above should be set to false (3d and not categorical)
plot_data = False
if plot_data:
    pp.plot_all_data(X_data, Y_data, sample_start=0, sample_end=3, slices_start=175, slices_end=180)

# splitting of the data
# training - testing
train_X, train_Y, test_X, test_Y = pp.generate_data(X_data, Y_data, 0.98)
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

# to recover the memory (delete original arrays)
del X_data, Y_data

# training - validation
train_X, train_Y, val_X, val_Y = pp.generate_data(train_X, train_Y, 0.9)

# sanity check for shape and number of samples
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
learn_rate = 2e-3
dropout_val = 0.2
img_dim = (train_X[0].shape[0], train_X[0].shape[1], train_X[0].shape[2], train_X[0].shape[3])
model_params_str = str(no_epochs) + '_' + str(b_size) + '_' + "{:.0e}".format(learn_rate) + '_' + str(dropout_val)

# model building and running (2 methods)
def model_train(model_type:str, 
                num_classes:int, 
                lr:float, 
                batch:int, 
                num_epochs:int, 
                dropout_value:float, 
                img_shape:tuple
                ): # tuple (h, w, d, c)

    if model_type == "unet_w_bbone":
        backbone = 'resnet101'
        activation_fun = 'softmax'
        decoder_val = 'upsampling'
        encode_weights = None

        opti = tf.keras.optimizers.RMSprop(learning_rate=lr)
        loss_fun = [sm.losses.DiceLoss(), sm.losses.CategoricalFocalLoss()]
        metric_fun = [sm.metrics.IOUScore(), sm.metrics.FScore()]

        sm.set_framework('tf.keras')
        model = sm.Unet(backbone, 
                        classes=num_classes,
                        input_shape= img_shape,
                        encoder_weights=encode_weights,
                        activation=activation_fun,
                        decoder_block_type=decoder_val,
                        # decoder_filters=(256, 128, 64, 32, 16),
                        # decoder_use_batchnorm=True,
                        dropout=dropout_value
                        )
        model.compile(optimizer = opti, loss=loss_fun, metrics=metric_fun)
        model.summary()
        print('\nStarting the training ...\n')
        history= model.fit(train_X, train_Y, epochs=num_epochs, batch_size=batch, validation_data=(val_X, val_Y))


    elif model_type == "unet_wo_bbone":
        deconv_type='upsampling',
        encoder_sizes=[16, 32, 64, 128, 256],
        decoder_sizes=[128, 64, 32, 16]
        opti = tf.keras.optimizers.RMSprop(learning_rate=lr)
        loss_fun = tf.keras.losses.CategoricalCrossentropy()
        metric_fun = [tf.keras.metrics.MeanIoU(num_classes=num_classes), 'accuracy']

        model = un3.build_unet(img_shape, 
                           no_classes, 
                           dropout_val, 
                           upconv_type=deconv_type, 
                           encode_filter_size=encoder_sizes,
                           decode_filter_size=decoder_sizes
                           )
        
        model.compile(optimizer = opti, loss=loss_fun, metrics=metric_fun)
        print('\nStarting the training ...\n')
        history = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), batch_size=batch, epochs=num_epochs)

    else:
        raise ValueError("Invalid string literal for model_type")
    
    return model, history

trained_model, trained_model_hist = model_train(model_type="unet_w_bbone", 
                                                num_classes=no_classes, 
                                                lr=learn_rate, 
                                                batch=b_size, 
                                                num_epochs=no_epochs, 
                                                dropout_value=dropout_val, 
                                                img_shape=img_dim
                                                )

# save the model and the history
po.save_model_hist(model_params_str, saved_models_path, trained_model, trained_model_hist)

# plot the losses and metrics
po.model_plots(trained_model_hist)

# predictions
test_Y_3d_vals, preds_cat_vals, preds_3d_vals = po.model_predictions(trained_model, test_X, test_Y, make_nifti=False)

# plot categorical (per class)
po.plot_truth_pred_cat(test_Y,
                       preds_cat_vals,
                       class_ind=0, 
                       sample_ind=0,
                       slice_start=175, 
                       slice_end=180
                       )

# plot 3d (not categorical)
po.plot_truth_pred_3d(test_Y_3d_vals,
                      preds_3d_vals,
                      sample_ind=0,
                      slice_start=175, 
                      slice_end=180
                      )

# load model
# loaded_model = po.model_load("unet_w_bbone", model_params_str, saved_models_path)

#@title
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