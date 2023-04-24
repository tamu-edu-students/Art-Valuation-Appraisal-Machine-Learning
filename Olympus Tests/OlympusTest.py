from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import argparse
import locale
import os

from sklearn.preprocessing import LabelEncoder

import numpy as np 
import pandas as pd
import tensorflow as tf 
import os
import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
import random
import time

imageID_Path = os.listdir('data/Modern_Contemporary')
imagePath_df = pd.DataFrame({'ID':[n.split('.')[0] for n in imageID_Path],
                                'Path':['data/Modern_Contemporary/' + n for n in imageID_Path]})

csv_dataPath = os.path.join('data/Modern_Contemporary','data.csv')
imageFeatures_df = pd.read_csv(csv_dataPath, sep=',')

# imageID[:}
# 

imagePath_df["ID"]= pd.to_numeric(imagePath_df["ID"],errors='coerce')

imagePath_df.reset_index(inplace=True)


for i,row in imagePath_df.iterrows():
    picture_Path = row['Path']
    # print(picture_Path)
    extCheck = len(picture_Path)
    path = picture_Path[extCheck-4:]
    if (path != 'JPEG'):
        imagePath_df.drop(i, inplace=True)


imagePath_df.drop(['index'], inplace=True, axis=1)

print(imagePath_df.head(),'\n')
# print(imageFeatures_df.head(), '\n')

print(len(imagePath_df))

# print(len(merged_df))

## Random Image 


merged_df = pd.merge(left=imagePath_df, right=imageFeatures_df, on='ID', how='inner', )

# merged_df = merged_df.drop('Size'
# merged_df = merged_df.drop(['Medium', 'Size' ],  axis=1, )


print(merged_df.head())
# print(merged_df.tail())
print(len(merged_df))



## Validating the data

for i, row in merged_df.iterrows():
    if(merged_df.loc[i, 'Hammer Price'] == 'Not Sold'): 
        merged_df.loc[i, 'Hammer Price'] = 0.0
    
merged_df['Hammer Price'] = pd.to_numeric(merged_df['Hammer Price'], errors='coerce')

merged_df['Low Estimation Price'] = merged_df['Low Estimation Price'].replace(np.nan, 0)







for i, row in merged_df.iterrows():
    mean = (row['Low Estimation Price'] + row['High Estimation Price']) / 2
    # means.append(mean)
    if(row['Hammer Price'] == 0.0): 
        merged_df.loc[i, 'Hammer Price'] = mean








# ##Creating List for Compressed Numpy array data paths
# npz_paths = []
images = []
id = []
IMAGE_RESIZE = (254,254)

# # merged_df = merged_df.reset_index(drop=True)

for i, row in merged_df.iterrows():

#     ## ASsigning iamge path the picture path
    picture_Path = row['Path']
    iD = row = row["ID"]

#     ## Loading Image from picture path
    pic_bgr_arr = cv2.imread(picture_Path)
    pic_rgb_arr = cv2.cvtColor(pic_bgr_arr, cv2.COLOR_BGR2RGB)
    pic_rgb_arr = cv2.resize(pic_rgb_arr, IMAGE_RESIZE)
    pic_rgb_arr = pic_rgb_arr/ 255.0


    images.append(pic_rgb_arr)
    id.append(iD)



dfData = list(zip(id, images))

images_df = pd.DataFrame(dfData, columns=['ID', 'Image'])


    

# print(merged_df.head())
# # print(merged_df.tail())



def get_X_y(df):

    dfArr = np.array(df)

    X_pic = []
    (y_artist, y_estimations, y_sale, y_medium) = [], [], [], []

    for i in range(len(df)): 

        pic = dfArr[i, 5]
        X_pic.append(pic)

        artist = dfArr[i,0]
        y_artist.append(artist)

        estimations = dfArr[i,1:3]
        y_estimations.append(estimations)

        sale = dfArr[i,3]
        y_sale.append(sale)

        medium = dfArr[i,4]
        y_medium.append(medium)

    X_pic = np.asarray(X_pic)

    Y_artist = np.asarray(y_artist)

    Y_estimations = np.asarray(y_estimations)

    Y_sale = np.asarray(y_sale)

    Y_medium = np.asarray(y_medium)


    return X_pic, (Y_artist, Y_estimations, Y_sale, Y_medium )





# inputs_df = merged_df
merged_df = pd.merge(left=merged_df, right=images_df, on='ID', how='inner', )




modelData_df.drop(['ID', 'Path','Size'], axis = 1, inplace=True)
modelData_df = modelData_df.replace(np.nan, 0)

modelData_df.head()

# ""'from sklearn.preprocessing import OneHotEncoder


# inputs_df.unique()


le = LabelEncoder()

modelData_df['Artist'] = le.fit_transform(modelData_df['Artist'])
modelData_df['Medium'] = le.fit_transform(modelData_df['Medium'])
# # artistLabels = np.array(artistLabels).ravel()


# artistLabels = ohe.categories_
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

# inputs_df.head()
print(modelData_df.dtypes)





maxLow = modelData_df['Low Estimation Price'].max()
lowStd = modelData_df['Low Estimation Price'].std()

maxHigh = modelData_df['High Estimation Price'].max()
highStd = modelData_df['High Estimation Price'].std()


maxSale = modelData_df['Hammer Price'].max()
saleStd = modelData_df['Hammer Price'].std()

for i in range(len(modelData_df)):
    modelData_df.loc[i, 'Low Estimation Price'] = modelData_df.at[i, 'Low Estimation Price'] / maxLow
    modelData_df.loc[i, 'High Estimation Price'] = modelData_df.at[i, 'High Estimation Price'] / maxHigh
    modelData_df.loc[i, 'Hammer Price'] = modelData_df.at[i, 'Hammer Price'] / maxSale









shuffled_df = modelData_df.sample(frac=1)

train_df, val_df, test_df = shuffled_df[:1440], shuffled_df[1440:2160], shuffled_df[2160:]



X_train_pic, (Y_train_art, Y_train_est, Y_train_sale, Y_train_med ) = get_X_y(train_df)

X_val_pic, (Y_val_art,    Y_val_est,     Y_val_sale,    Y_val_med ) = get_X_y(val_df)

X_test_pic,  (Y_test_art,    Y_test_est,     Y_test_sale,    Y_test_med ) = get_X_y(test_df)






## Define the CNN Model


inputImage = Input(shape=( 254,254,3), name='Art Image Input')

x = Conv2D(256, 3, activation='relu')(inputImage)
x = Conv2D(128, 3, activation='relu')(x)

OutputBlock0 = MaxPooling2D(4)(x)

x = Conv2D(128, 3, activation='relu')(OutputBlock0)
x = Conv2D(64, 3, activation='relu')(x)

OutputBlock1 = MaxPooling2D(3)(x)

x = Conv2D(64, 3, activation='relu', padding='same')(OutputBlock1)
x = Conv2D(64, 3, activation='relu', padding='same')(x)

OutputBlock2 = layers.add([x, OutputBlock1])

x = Conv2D(32, 3, activation='relu', padding='same')(OutputBlock2)
x = Conv2D(32, 3, activation='relu', padding='same')(x)

OutputBlock3 = MaxPooling2D()(x)

x = Conv2D(32, 3, activation='relu', padding='same')(OutputBlock3)
x = Conv2D(32, 3, activation='relu', padding='same')(x)

x = layers.GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

output1 = Dense(1, activation='sigmoid', name='Artist')(x)
output2 = Dense(2, activation='relu', name='Estimaitions')(x)
output3 = Dense(1, activation='relu', name='Sale')(x)
output4 = Dense(1, activation='sigmoid', name='Medium')(x)




# z = layers.Dense(1, activation='linear')(z)

model = Model(inputs=[inputImage] ,outputs=[output1, output2, output3, output4], name="ArtValuationAI")




model.summary()


# dot_img_file = '/Multi_Output_Model.png'
# tf.keras.utils.plot_model(model, dot_img_file,show_shapes=True)


optimzer = Adam(learning_rate=0.001)
model.compile(loss=['categorical_crossentropy', 'mse', 'mse', 'categorical_crossentropy'], optimizer=optimzer, metrics=[ 'accuracy'])

cp = ModelCheckpoint('model/', save_best_only=True)


logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)




X_train_picTensor = tf.convert_to_tensor(X_train_pic, dtype="float64")

Y_train_artTensor = tf.convert_to_tensor(Y_train_art, dtype="int32")
Y_train_estTensor = tf.convert_to_tensor(Y_train_est, dtype="float64")
Y_train_saleTensor = tf.convert_to_tensor(Y_train_sale, dtype="float64")
Y_train_medTensor = tf.convert_to_tensor(Y_train_med, dtype="int32")


X_val_picTensor = tf.convert_to_tensor(X_val_pic, dtype="float64")

Y_val_artTensor = tf.convert_to_tensor(Y_val_art, dtype="int32")
Y_val_estTensor = tf.convert_to_tensor(Y_val_est, dtype="float64")
Y_val_saleTensor = tf.convert_to_tensor(Y_val_sale, dtype="float64")
Y_val_medTensor = tf.convert_to_tensor(Y_val_med, dtype="int32")

X_test_picTensor =  tf.convert_to_tensor(X_test_pic, dtype="float32")

Y_test_artTensor =  tf.convert_to_tensor(Y_test_art, dtype="int32")
Y_test_estTensor = tf.convert_to_tensor(Y_test_est, dtype="float64")
Y_test_saleTensor = tf.convert_to_tensor(Y_test_sale, dtype="float64")
Y_test_medTensor = tf.convert_to_tensor(Y_test_med, dtype="int32")

hist = model.fit(x=X_train_picTensor, y=[Y_train_artTensor, Y_train_estTensor, Y_train_saleTensor, Y_train_medTensor],
          validation_data=(X_val_picTensor, [Y_val_artTensor, Y_val_estTensor, Y_val_saleTensor, Y_val_medTensor]), 
          callbacks=[tensorboard_callback,cp], 
          epochs=1 )