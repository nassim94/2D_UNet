import keras
from keras.models import Model
from keras.layers import Activation
from keras.layers import concatenate, Conv2D, MaxPooling2D
from keras.layers import Input, merge, UpSampling2D,BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from metrics import *


LR = 1e-4
def unet(input_size=(240, 240, 1)):    #I tried different input size
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    batch1 = BatchNormalization()(conv1)
    batch1 = Activation('relu')(batch1)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(batch1)
    batch1 = BatchNormalization()(conv1)
    batch1 = Activation('relu')(batch1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)
    pool1 = Dropout(0.25)(pool1)
    ####
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    batch2 = BatchNormalization()(conv2)
    batch2 = Activation('relu')(batch2)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(batch2)
    batch2 = BatchNormalization()(conv2)
    batch2 = Activation('relu')(batch2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch2)
    pool2 = Dropout(0.5)(pool2)
    #####
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    batch3 = BatchNormalization()(conv3)
    batch3 = Activation('relu')(batch3)
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(batch3)
    batch3 = BatchNormalization()(conv3)
    batch3 = Activation('relu')(batch3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(batch3)
    pool3 = Dropout(0.5)(pool3)
    ####
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    batch4 = BatchNormalization()(conv4)
    batch4 = Activation('relu')(batch4)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(batch4)
    batch4 = BatchNormalization()(conv4)
    batch4 = Activation('relu')(batch4)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(batch4))

    merge7 = concatenate([batch3, up7], axis=3)
    merge7 = Dropout(0.5)(merge7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
    batch7 = BatchNormalization()(conv7)
    batch7 = Activation('relu')(batch7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(batch7)
    batch7 = BatchNormalization()(conv7)
    batch7 = Activation('relu')(batch7)
    ######
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(batch7))

    merge8 = concatenate([batch2, up8], axis=3)
    merge8 = Dropout(0.5)(merge8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
    batch8 = BatchNormalization()(conv8)
    batch8 = Activation('relu')(batch8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(batch8)
    batch8 = BatchNormalization()(conv8)
    batch8 = Activation('relu')(batch8)
    ###########
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(batch8))

    merge9 = concatenate([batch1, up9], axis=3)
    merge9 = Dropout(0.5)(merge9)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
    batch9 = BatchNormalization()(conv9)
    batch9 = Activation('relu')(batch9)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(batch9)
    batch9 = BatchNormalization()(conv9)
    batch9 = Activation('relu')(batch9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(batch9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(learning_rate=LR), loss=combined_loss,
                  metrics=[dice_coef()])

    return model