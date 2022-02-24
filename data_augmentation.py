import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2

datagen = ImageDataGenerator(rotation_range =15, 
                     width_shift_range = 0.2, 
                     height_shift_range = 0.2,  
                     rescale=1./255, 
                     shear_range=0.2, 
                     zoom_range=0.2, 
                     horizontal_flip = True, 
                     fill_mode = 'nearest', 
                     data_format='channels_last', 
                     brightness_range=[0.5, 1.5]) 

imgs = os.listdir("/media/patient/01/augment/Data_Augmentation/Data")

for img in imgs:
    img=cv2.imread("/media/patient/01/augment/Data_Augmentation/Data"+"/"+img)
    print(img)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow (x, batch_size=1, save_to_dir =r'/media/patient/01/augment/Data_Augmentation/R', save_prefix ='', save_format='jpg'):
        i+=1
        if i>10:
            break
