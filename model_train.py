# model building and training (two methods)

# imports
import numpy as np
import tensorflow as tf
import segmentation_models_3D as sm

def model_train(train_val_data:tuple, 
                model_type:str,
                num_classes:int, 
                lr:float, 
                batch:int, 
                num_epochs:int, 
                dropout_value:float, 
                img_shape:tuple,
                shuffle_bool:bool
                ): # tuple (h, w, d, c)
    """
    Performs the model training depending on the chosen method (transfer learning (1) or unet from scratch)

    Inputs: 
        train_val_data: training and validation datasets
        model_type: unet with backbone for the encode part or unet from scratch
        num_classes: the number of classes in each sample
        lr: learning rate
        batch: batch size for training
        num_epochs: number of epochs for training the models
        dropout_value: value to be set for the dropout layer
        img_shape: shape of each ct-scan (or a mask)
        shufffle_bool: enable or disable shuffling while training

    Outputs:
        returns the trained model and its corresponding history

    """
    
    train_X, train_Y, val_X, val_Y = train_val_data
    if model_type == "unet_w_bbone":
        backbone = 'densenet201'
        activation_fun = 'softmax'
        decoder_val = 'transpose'
        encode_weights = None

        opti = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_fun = [sm.losses.DiceLoss(class_weights=np.array([0, 0.25, 0.25, 0.25, 0.25])), sm.losses.CategoricalFocalLoss()]
        metric_fun = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        sm.set_framework('tf.keras')
        model = sm.Unet(backbone, 
                        classes=num_classes,
                        input_shape= img_shape,
                        encoder_weights=encode_weights,
                        activation=activation_fun,
                        decoder_block_type=decoder_val,
                        # decoder_filters=(256, 128, 64, 32, 16),
                        decoder_filters=(512, 256, 128, 64, 32),
                        # decoder_use_batchnorm=True,
                        dropout=dropout_value
                        )
        model.compile(optimizer = opti, loss=loss_fun, metrics=metric_fun)
        model.summary()
        print('\nStarting the training ...\n')
        history= model.fit(train_X, train_Y, epochs=num_epochs, batch_size=batch, validation_data=(val_X, val_Y), shuffle=shuffle_bool)

    # second
    elif model_type == "unet_wo_bbone":
        deconv_type='transpose'
        encode_sizes=[16, 32, 64, 128, 256]
        decode_sizes=[128, 64, 32, 16]
        opti = tf.keras.optimizers.Adam(learning_rate=lr) # optimizer
        loss_fun = tf.keras.losses.CategoricalCrossentropy()
        metric_fun = [tf.keras.metrics.MeanIoU(num_classes=num_classes)]

        model = un3.build_unet(img_shape, 
                               no_classes, 
                               dropout_val, 
                               upconv_type=deconv_type, 
                               encode_filter_size=encode_sizes,
                               decode_filter_size=decode_sizes
                               )
        
        model.compile(optimizer = opti, loss=loss_fun, metrics=metric_fun)
        print('\nStarting the training ...\n')
        history = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), batch_size=batch, epochs=num_epochs, shuffle=shuffle_bool)

    else:
        raise ValueError("Invalid string literal for model_type")
    
    return model, history