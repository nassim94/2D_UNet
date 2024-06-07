import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from skimage.transform import resize
from plotting import *
from UNet import *
################### test data #######################
os.chdir('/content/img....')
lst_test = os.listdir('/content/img....')

img_test = []
for filename in lst_test:
   img_test.append(filename)
img_test.sort()
##############
os.chdir('/content/mask....')
lst_mask  = os.listdir('/content/mask....')


mask_t = []
for filename in lst_mask:
   mask_t.append(filename)
mask_t.sort()
############################
X_TEST = np.zeros((2994,240, 240, 1), dtype=np.float32)
n = -1
for name in img_test:
    n = n+1
    image = img_to_array(load_img("/content/img.../"+name, color_mode = "grayscale"))
    image = resize(image, (240, 240, 1), mode = 'constant', preserve_range = True)
    # Save images
    X_TEST[n] = image/255.0

y_test = np.zeros((2994, 240,240, 1), dtype=np.float32)
n = -1
for name in mask_t:
    n = n+1
    image = img_to_array(load_img("/content/mask.../"+name, color_mode = "grayscale"))
    image = resize(image, (240, 240, 1), mode = 'constant', preserve_range = True)
    # Save images
    y_test[n] = image/255.0
#########################################################
#Load train data
seed = 1

image_datagen = ImageDataGenerator(
    #vertical_flip=True,
    #horizontal_flip=True,
    #rotation_range=..
    #.....
    rescale=1./255
)
mask_datagen = ImageDataGenerator(
    # vertical_flip=True,
    # horizontal_flip=True,
    # rotation_range=..
    # .....
    rescale=1./255
)
image_generator = image_datagen.flow_from_directory(
        '/content/train',
        classes = ['img'],
        class_mode = None,
        color_mode = 'grayscale',
        target_size = (240, 240),
        batch_size = 16,
        save_to_dir = None,
        seed = seed)
mask_generator = mask_datagen.flow_from_directory(
        '/content/train',
        classes = ['mask'],
        class_mode = None,
        color_mode = 'grayscale',
        target_size = (240, 240),
        batch_size = 16,
        save_to_dir = None,
        seed = seed)
train_generator = zip(image_generator, mask_generator)

###################### validation
seed = 2
image1_datagen = ImageDataGenerator(rescale=1./255)
mask1_datagen = ImageDataGenerator(rescale=1./255)


image1_generator = image1_datagen.flow_from_directory(
        '/content/valid',
        classes = ['img'],
        class_mode = None,
        color_mode = 'grayscale',
        target_size = (240, 240),
        batch_size = 16,
        save_to_dir = None,
        seed = seed)
mask1_generator = mask1_datagen.flow_from_directory(
        '/content/valid',
        classes = ['mask'],
        class_mode = None,
        color_mode = 'grayscale',
        target_size = (240, 240),
        batch_size = 16,
        save_to_dir = None,
        seed = seed)
valid_generator = zip(image1_generator, mask1_generator)

######## define model
model = unet()
######################### train data
model_checkpoint = ModelCheckpoint('/content/*.hdf5', monitor='loss',verbose=1, save_best_only=True)

earlystopper = EarlyStopping(patience=3, verbose=1,monitor='loss')

num_valid_data = *
num_train_data = *
epoch_num = *
valid_step = num_valid_data/16
train_step = num_train_data/16

history = model.fit(train_generator, steps_per_epoch=train_step, epochs=epoch_num,
                    validation_data=valid_generator, callbacks = [model_checkpoint,earlystopper], validation_steps=valid_step)
################# evaluate model on test data
model.evaluate(X_TEST, y_test, verbose=1)
pred = model.predict(X_TEST)

#################### plot some model outputs

th = *
preds_val_t = (pred > th).astype(np.uint8)
plot_sample(X_TEST, y_test,preds_val_t, ix=1)