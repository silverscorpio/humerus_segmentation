# imports
from tensorflow.keras.layers import Conv3D, BatchNormalization, MaxPooling3D, Input, UpSampling3D, Concatenate, Dropout, Conv3DTranspose
from tensorflow.keras import Model

# function to build and return the unet model
def build_unet(img_shape:tuple, num_classes:int, dropout:float, upconv_type:str, encode_filter_size:list, decode_filter_size:list):
    """

    Builds and returns the Unet architecture/model based on the Ronneberger Unet paper for medical image segmentation
    using the given inputs

    Inputs:
        ct_scan/mask shape
        number of classes
        dropout layer value
        decoder part deconvolution types: (upsampling or Conv3dTranspose)
        encoder filters sizes
        decoder filters sizes

    Outputs:
        built Unet model

    """

    if len(encode_filter_size) != 5 or len(decode_filter_size) != 4:
        raise ValueError("incorrect number of values for encoder & decoder blocks (encoder:5, decoder:4)")
    
    kernel_initializer =  'he_uniform'
    inputs = Input(img_shape)
    s = inputs

    #Contraction path (encoder)
    # encoder 1
    c1 = Conv3D(encode_filter_size[0], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s) #16
    c1 = Dropout(dropout)(c1)
    c1 = Conv3D(encode_filter_size[0], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1) # 16
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    # encoder 2
    c2 = Conv3D(encode_filter_size[1], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1) # 32
    c2 = Dropout(dropout)(c2)
    c2 = Conv3D(encode_filter_size[1], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2) # 32
    p2 = MaxPooling3D((2, 2, 2))(c2)
    
    # encoder 3
    c3 = Conv3D(encode_filter_size[2], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2) # 64
    c3 = Dropout(dropout)(c3)
    c3 = Conv3D(encode_filter_size[2], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3) # 64
    p3 = MaxPooling3D((2, 2, 2))(c3)
    
    # encoder 4
    c4 = Conv3D(encode_filter_size[3], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3) # 128
    c4 = Dropout(dropout)(c4)
    c4 = Conv3D(encode_filter_size[3], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4) # 128
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
    
    # bridge
    c5 = Conv3D(encode_filter_size[4], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4) # 256
    c5 = Dropout(dropout)(c5)
    c5 = Conv3D(encode_filter_size[4], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5) # 256
    
    # expansion path (decoder)
    # decoder 1
    if upconv_type == "upsampling":
        u6 = UpSampling3D(size=(2,2,2))(c5)
    elif upconv_type == "transpose":
        u6 = Conv3DTranspose(decode_filter_size[0], (2, 2, 2), strides=(2, 2, 2), padding='same')(c5) # 128
    u6 = Concatenate()([u6, c4])
    c6 = Conv3D(decode_filter_size[0], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6) # 128
    c6 = Dropout(dropout)(c6)
    c6 = Conv3D(decode_filter_size[0], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6) # 128
    
    # decoder 2
    if upconv_type == "upsampling":
        u7 = UpSampling3D(size=(2,2,2))(c6)
    elif upconv_type == "transpose":
        u7 = Conv3DTranspose(decode_filter_size[1], (2, 2, 2), strides=(2, 2, 2), padding='same')(c6) # 64
    u7 = Concatenate()([u7, c3])
    c7 = Conv3D(decode_filter_size[1], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7) # 64
    c7 = Dropout(dropout)(c7)
    c7 = Conv3D(decode_filter_size[1], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7) # 64
    
    # decoder 3
    if upconv_type == "upsampling":
        u8 = UpSampling3D(size=(2,2,2))(c7)
    elif upconv_type == "transpose":
        u8 = Conv3DTranspose(decode_filter_size[2], (2, 2, 2), strides=(2, 2, 2), padding='same')(c7) # 32
    u8 = Concatenate()([u8, c2])
    c8 = Conv3D(decode_filter_size[2], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8) # 32
    c8 = Dropout(dropout)(c8)
    c8 = Conv3D(decode_filter_size[2], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8) # 32
    
    # decoder 4
    if upconv_type == "upsampling":
        u9 = UpSampling3D(size=(2,2,2))(c8)
    elif upconv_type == "transpose":
        u9 = Conv3DTranspose(decode_filter_size[3], (2, 2, 2), strides=(2, 2, 2), padding='same')(c8) # 16
    u9 = Concatenate()([u9, c1])
    c9 = Conv3D(decode_filter_size[3], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9) # 16
    c9 = Dropout(dropout)(c9)
    c9 = Conv3D(decode_filter_size[3], (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9) # 16
    
    # output
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
    
    # model construction
    built_model = Model(inputs=[inputs], outputs=[outputs])
    built_model.summary()
    
    return built_model