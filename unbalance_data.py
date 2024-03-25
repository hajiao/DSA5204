import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf
import os
import pickle

sns.set(font_scale=1.5, style='darkgrid')

tf.config.list_physical_devices('GPU')
# tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)
    # logical_gpus = tf.config.list_logical_devices('GPU')
    # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
    
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


cat_indices = np.where(y_train == 3)[0]
dog_indices = np.where(y_train == 5)[0]
print(f"Number of cat samples: {len(cat_indices)}")
print(f"Number of dog samples: {len(dog_indices)}")

# Define the imbalance ratio for the "dog" class relative to the "cat" class
ratio = 9
imbalance_ratio = 0.1 * ratio

num_epochs = 100


# Select a random subset of the "cat" class with size equal to the "dog" class times the imbalance ratio
cat_indices = np.where(y_train == 3)[0]
num_cat_samples = int(len(cat_indices) * imbalance_ratio)
cat_subset_indices = np.random.choice(cat_indices, size=num_cat_samples, replace=False)
cat_indices_test = np.where(y_test == 3)[0]
num_cat_samples_test = int(len(cat_indices_test) * imbalance_ratio)
cat_subset_indices_test = np.random.choice(cat_indices_test, size=num_cat_samples_test, replace=False)


# Select a random subset of the "dog" class with size equal to the number of samples in the "cat" subset
dog_indices = np.where(y_train == 5)[0]
dog_indices_test = np.where(y_test == 5)[0]
# dog_subset_indices = np.random.choice(dog_indices, size=len(cat_subset_indices), replace=False)

# Combine the subsets to create the unbalanced dataset
subset_indices = np.concatenate((cat_subset_indices, dog_indices))
x_train_subset = x_train[subset_indices]
y_train_subset = y_train[subset_indices]
x_train_subset_1 = x_train[cat_subset_indices]
x_train_subset_2 = x_train[dog_indices]
y_train_subset_1 = y_train[cat_subset_indices]
y_train_subset_2 = y_train[dog_indices]
subset_indices_test = np.concatenate((cat_subset_indices_test, dog_indices_test))
x_test_subset = x_test[subset_indices_test]
y_test_subset = y_test[subset_indices_test]
x_test_subset_1 = x_test[cat_subset_indices_test]
x_test_subset_2 = x_test[dog_indices_test]
y_test_subset_1 = y_test[cat_subset_indices_test]
y_test_subset_2 = y_test[dog_indices_test]


y_train_subset = np.where(y_train_subset == 5, 1, 0)
y_train_subset_1 = np.where(y_train_subset_1 == 5, 1, 0)
y_train_subset_2 = np.where(y_train_subset_2 == 5, 1, 0)
y_test_subset = np.where(y_test_subset == 5, 1, 0)
y_test_subset_1 = np.where(y_test_subset_1 == 5, 1, 0)
y_test_subset_2 = np.where(y_test_subset_2 == 5, 1, 0)

# Generate a random permutation of the indices
indices = np.random.permutation(len(x_train_subset))

# Use the permutation to shuffle the arrays
x_train_subset = x_train_subset[indices]
y_train_subset = y_train_subset[indices]


# Verify the class imbalance in the subset
unique, counts = np.unique(y_train_subset, return_counts=True)
print(dict(zip(unique, counts)))
unique, counts = np.unique(y_test_subset, return_counts=True)
print(dict(zip(unique, counts)))

print(y_train_subset[0])

from tensorflow.keras.utils import to_categorical

#change the label to one-hot coding
y_train_subset=tf.keras.utils.to_categorical(y_train_subset)
y_test_subset=tf.keras.utils.to_categorical(y_test_subset)
y_train_subset_1 = to_categorical(y_train_subset_1, num_classes=2)
y_train_subset_2 = to_categorical(y_train_subset_2, num_classes=2)
y_test_subset_1 = to_categorical(y_test_subset_1, num_classes=2)
y_test_subset_2 = to_categorical(y_test_subset_2, num_classes=2)

print(y_train_subset[0])
print(y_test_subset[0])
print(y_train_subset_1[0])
print(y_train_subset_2[0])
print(y_test_subset_1[0])
print(y_test_subset_2[0])

from dsa5204_project_utils import *
tf.__version__

use_mixed_float16=False
#learning related configurations
lr_base=5e-4 #0.016
lr_min=0
lr_decay_epoch=2.4
lr_decay_factor=0.97
batch_size=1024 #4096
scaled_lr=lr_base*batch_size/256.
scaled_lr_min=lr_min*batch_size/256.
steps_per_epoch=500

AUTOTUNE = tf.data.AUTOTUNE
train_ds=tf.data.Dataset.from_tensor_slices(dict(images=x_train,labels=tf.cast(y_train,tf.float32)))
valid_ds=train_ds.take(5000).batch(batch_size)
valid_ds=valid_ds.map(preprocess_for_model, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

'''
Some configuration used in model construction
'''
act=keras.activations.gelu
depth_divisor=8
min_depth=8
survival_prob=0.8
dropout=0.2
nclasses=10
conv_dropout=None

#b0_config
v2_base_block = [ # The baseline config for v2 models.
'r1_k3_s1_e1_i32_o16_c1',
'r2_k3_s2_e4_i16_o32_c1',
'r2_k3_s2_e4_i32_o48_c1',
'r3_k3_s2_e4_i48_o96_se0.25',
'r5_k3_s1_e6_i96_o112_se0.25',
'r8_k3_s2_e6_i112_o192_se0.25',
]

v2_base_block_modified2 = [ # The baseline config for v2 models.
'r1_k3_s1_e1_i32_o16_c1',
'r2_k3_s2_e4_i16_o32_c1',
'r2_k3_s1_e4_i32_o48_c1',
'r3_k3_s1_e4_i48_o96_se0.25',
'r5_k3_s1_e6_i96_o112_se0.25',
'r8_k3_s2_e6_i112_o192_se0.25',
]

v2_base_block_modified = [ # The baseline config for v2 models.
'r1_k3_s1_e1_i32_o16_c1',
'r1_k3_s1_e4_i16_o32_c1',
'r1_k3_s1_e4_i32_o48_c1',
'r2_k3_s2_e4_i48_o96_se0.25',
'r2_k3_s1_e6_i96_o112_se0.25',
'r5_k3_s2_e6_i112_o192_se0.25',
]

'''
structure modified from ConvNeXt:
    1:1:3:1 stage ratio, 
    7x7 kernel, 
    more filters, 
    only use depthwise convolution
    
'''
v2_base_block_modified3 = [
'r2_k5_s2_e4_i12_o24_c1',
'r2_k5_s1_e4_i24_o32_c1',
'r6_k5_s2_e4_i32_o48_se0.25_c1',
'r2_k5_s1_e4_i48_o96_se0.25_c1',
]


keras.mixed_precision.set_global_policy('float32')
if use_mixed_float16:
  keras.mixed_precision.set_global_policy('mixed_float16')

'''
EfficientNetV2 backbone construct
'''
efficientv2_base=EfficientV2base(act,decode_cfgs(v2_base_block_modified),dropout = 0.5)
'''
Adding input and output (classfication) layer to get a trainable (usable) model
'''
inputx=keras.Input((None,None,3))
x=efficientv2_base(inputx)
x=keras.layers.Dense(2,activation='softmax')(x)
Mymodel=keras.Model(inputs=inputx,outputs=x)

from keras.optimizers import Adam

Mymodel.compile(optimizer = Adam(),
                loss='binary_crossentropy',
                metrics=[keras.metrics.BinaryAccuracy(name='acc')])


import numpy as np
import matplotlib.pyplot as plt

print(np.shape(x_train_subset))

# Define empty lists to store the accuracy for each category
train_acc_record = []
test_acc_record = []
cat1_train_acc = []
cat2_train_acc = []
cat1_test_acc = []
cat2_test_acc = []

# Iterate over the specified number of epochs
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))

    # Train the model for one epoch
    history = Mymodel.fit(x_train_subset, y_train_subset,
                      batch_size=1024,
                      epochs=1,
                      verbose=1,
                      validation_data=(x_test_subset, y_test_subset))

    # Evaluate the model on the train set for category 1
    train_loss, train_acc = Mymodel.evaluate(x_train_subset_1, y_train_subset_1, verbose=0)
    cat1_train_acc.append(train_acc)

    # Evaluate the model on the test set for category 1
    test_loss, test_acc = Mymodel.evaluate(x_test_subset_1, y_test_subset_1, verbose=0)
    cat1_test_acc.append(test_acc)

    # Evaluate the model on the train set for category 2
    train_loss, train_acc = Mymodel.evaluate(x_train_subset_2, y_train_subset_2, verbose=0)
    cat2_train_acc.append(train_acc)

    # Evaluate the model on the test set for category 2
    test_loss, test_acc = Mymodel.evaluate(x_test_subset_2, y_test_subset_2, verbose=0)
    cat2_test_acc.append(test_acc)
    
    # Record the training history
    train_acc_record.append(history.history['acc'])
    test_acc_record.append(history.history['val_acc'])
    
    
    
    

    # Print the train and test accuracies for each category
    print('Category 1 train accuracy: {:.4f}'.format(cat1_train_acc[-1]))
    print('Category 2 train accuracy: {:.4f}'.format(cat2_train_acc[-1]))
    print('Category 1 test accuracy: {:.4f}'.format(cat1_test_acc[-1]))
    print('Category 2 test accuracy: {:.4f}'.format(cat2_test_acc[-1]))
    
    # with open('history_%d.pickle' % ratio, 'wb') as f:
      # pickle.dump(history.history, f)
    
    np.save('output/cat1_train_acc_%d.npy' % ratio, cat1_train_acc)
    np.save('output/cat2_train_acc_%d.npy' % ratio, cat2_train_acc)
    np.save('output/train_acc_%d.npy' % ratio, train_acc_record)
    np.save('output/cat1_test_acc_%d.npy' % ratio, cat1_test_acc)
    np.save('output/cat2_test_acc_%d.npy' % ratio, cat2_test_acc)
    np.save('output/test_acc_%d.npy' % ratio, test_acc_record)

    
    Mymodel.save_weights('model_weights_%d.h5' % ratio)