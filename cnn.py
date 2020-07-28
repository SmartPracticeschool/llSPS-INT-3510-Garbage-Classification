import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
trainDataGen = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2, shear_range = 0.2, horizontal_flip = True)
testDataGen = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2, shear_range = 0.2, horizontal_flip = True)

path='./Dataset/Garbage classification/Garbage classification'

trainX = trainDataGen.flow_from_directory('./Dataset/Garbage classification', target_size=(64,64), batch_size=32, class_mode='categorical')
testX = testDataGen.flow_from_directory('./Dataset/Garbage classification', target_size=(64,64), batch_size=32, class_mode='categorical')

print(trainX.class_indices)



