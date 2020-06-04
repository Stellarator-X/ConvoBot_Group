# This will probably work better as a notebook

from model import *
import tensorflow as tf 

"""
Loading the Data
"""

"""
Data Generation
"""
datagen = None
# First batch with samples in increasing order
firstset = None

"""
Data Augmentation
"""

# Building the model
input_shape = None

model = DSModel(input_shape)
model.build()
model.compile()

model.summary()

# Training with sortagrad
hist1 = model.fit(firstset, epochs = 1)
hist2 = model.fit(datagen, epochs = 10, validation_data = None)