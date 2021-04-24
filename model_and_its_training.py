import os
import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

base_dir ='F:/computer vision projects/skin cancer/data/New folder/'

train_dir = os.path.join(base_dir,'train')
# test_dir = os.path.join(base_dir,'test')
validation_dir = os.path.join(base_dir,'valid')

train_melanoma_dir = os.path.join(train_dir,'melanoma')
train_nevus_dir = os.path.join(train_dir,'nevus')
train_seborrheic_keratosis_dir = os.path.join(train_dir,'seborrheic_keratosis')


valid_melanoma_dir = os.path.join(validation_dir,'melanoma')
valid_nevus_dir = os.path.join(validation_dir,'nevus')
valid_seborrheic_keratosis_dir = os.path.join(validation_dir,'seborrheic_keratosis')


num_melanoma_training = len(os.listdir(train_melanoma_dir))
num_nevus_training = len(os.listdir(train_nevus_dir))
num_seborrheic_keratosis_training = len(os.listdir(train_seborrheic_keratosis_dir))


num_melanoma_valid= len(os.listdir(valid_melanoma_dir))
num_nevus_valid = len(os.listdir(valid_nevus_dir))
num_seborrheic_keratosis_valid = len(os.listdir(valid_seborrheic_keratosis_dir))




total_training = num_melanoma_training + num_nevus_training + num_seborrheic_keratosis_training
total_validation = num_melanoma_valid + num_nevus_valid + num_seborrheic_keratosis_valid

print('total training melanoma images:', num_melanoma_training)
print('total training nevus images:', num_nevus_training)
print('total training seborrheic keratosis images:', num_seborrheic_keratosis_training)
print("--")

print('total validation melanoma images:', num_melanoma_valid)
print('total validation nevus images:', num_nevus_valid)
print('total validation seborrheic keratosis images:', num_seborrheic_keratosis_valid)
print("--")

print("Total training images:", total_training)
print("Total validation images:", total_validation)

train_examples = total_training
validation_examples = total_validation

img_height = img_size =img_width = 150
batch_size = 32


DESIRED_ACCURACY = 0.85


# Create an instance of the inception model from the local pre-trained weights
path_inception = f"./transfer_learning/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

local_weights_file = path_inception

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

# Print the model summary
pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
#calback = myCallback()
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024,activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(3, activation='softmax')(x)

model = Model( pre_trained_model.input, x)

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ["acc"])

model.summary()






# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   rotation_range=40,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                  )

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150,150)
                                                   )

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150,150)
                                                   )









class myCallback(tf.keras.callbacks.Callback):
    # Your Code
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get("acc") > DESIRED_ACCURACY):
            self.stop_traning = True


callbacks = myCallback()




history = model.fit_generator(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[callbacks],
    verbose=1
)


export_path_sm = "./models/{}.h5".format("skin_cancer_using_inception_v3_weights")
print(export_path_sm)
tf.saved_model.save(model, export_path_sm)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()

