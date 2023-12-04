from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, Sequential
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from trainingmonitor import TrainingMonitor
from matplotlib import pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import pandas as pd
import numpy as np 
import random
import cv2
import os



## Data Directory Path
print('#1 - Initialzing DataFrame...\n')
imageID_Path = os.listdir('data/main/')

## Creating data frame based on directory contents 
imagePath_df = pd.DataFrame({'ID':[n.split('.')[0] for n in imageID_Path],
                                'Path':['data/main/' + n for n in imageID_Path]})

## Extracting features of images from the CSV file
csv_dataPath = os.path.join('data/main/','data.csv')
imageFeatures_df = pd.read_csv(csv_dataPath, sep=',')

## COnvoerting all values to integer values
imagePath_df["ID"]= pd.to_numeric(imagePath_df["ID"],errors='coerce')



## Error handling for none JPEG Images

for i,row in imagePath_df.iterrows():
    picture_Path = row['Path']
    # print(picture_Path)
    extCheck = len(picture_Path)
    path = picture_Path[extCheck-4:]
    if (path != 'JPEG'):
        print("Path Dropped:", path)
        imagePath_df.drop(i, inplace=True)


print( "Numbers of Images succefully imported:",len(imagePath_df))

merged_df = pd.merge(left=imagePath_df, right=imageFeatures_df, on='ID', how='inner', )

print('Merging Dataframes...')








## Validating the data
print('Cleaning Data...')


merged_df['Hammer Price'] = pd.to_numeric(merged_df['Hammer Price'], errors='coerce')

merged_df = merged_df.replace(np.nan, 0)

# merged_df['Low Estimation Price'].replace(np.nan, 0)

means = []

for i, row in merged_df.iterrows():

    if(row['Low Estimation Price'] == 0 and row['High Estimation Price'] == 0):
        mean = row['Hammer Price'] / 2
    
    elif (row['Low Estimation Price'] == 0):
        mean = row['High Estimation Price'] / 2
    else:
        mean = (row['Low Estimation Price'] + row['High Estimation Price']) / 2
    means.append(mean)
    if(row['Hammer Price'] == 0.0): 
        merged_df.loc[i, 'Hammer Price'] = mean


# 
merged_df['Mean Estimation'] = pd.Series(means)









print('Normalizing Data...')

for i in range(len(merged_df)):
        picture_Path = merged_df.at[i,'Path']
        if not os.path.exists(picture_Path):
                print("Oops! File gone on vacation:", picture_Path)








# ##Creating List for Compressed Numpy array data paths
import os.path 

test_df = merged_df

print("Loading Images into Dataframe...")
images = []
id = []
IMAGE_RESIZE = (256,256)


for i, row in merged_df.iterrows():

    ## ASsigning iamge path the picture path
    picture_Path = row['Path']
    iD = row = row["ID"]

    if os.path.exists(picture_Path):
        # Reading Image Path, Color Correcting, and then resizing
        pic_bgr_arr = cv2.imread(picture_Path)
        
        if(pic_bgr_arr is not None):
            pic_rgb_arr = cv2.cvtColor(pic_bgr_arr, cv2.COLOR_BGR2RGB)
            pic_rgb_arr = cv2.resize(pic_rgb_arr, IMAGE_RESIZE)
            pic_rgb_arr = pic_rgb_arr / 255.0
        else:
            print("Error with Image", picture_Path)
            os.remove(picture_Path)
            merged_df.drop(i, inplace=True)

        # Append to Image List
        images.append(pic_rgb_arr)
        id.append(iD)
    
    else:
        continue
       

# ZIP contents to load into datafram
dfData = list(zip(id, images))
images_df = pd.DataFrame(dfData, columns=['ID', 'Image'])














# // merging with dataframe
merged_df = pd.merge(left=merged_df, right=images_df, on='ID', how='inner')

print("Label Encoding Artists...")


le = LabelEncoder()

merged_df['Artist'] = le.fit_transform(merged_df['Artist'])
artistLabels = le.classes_














merged_df.drop(['Low Estimation Price', 'High Estimation Price', 'Hammer Price', 'Size', 'Path', 'ID', 'Medium' , 'Mean Estimation' ],inplace=True, axis=1)




shuffled_df = merged_df.sample(frac=1)
print("Partitioning data...")
train_df, val_df, test_df = shuffled_df[:3500], shuffled_df[3500:4000], shuffled_df[4000:]

len(train_df), len(val_df), len(test_df)



print(merged_df.shape)







def get_X_y(df):

    dfArr = np.array(df)

    x_pic = []
    y_artist = []

    for i in range(len(df)): 

        pic = dfArr[i, 1]
        x_pic.append(pic)

        artist = dfArr[i, 0]
        y_artist.append(artist)

    X_pic = np.asarray(x_pic)
    y_artist = np.asarray(y_artist)


    return X_pic, y_artist

X_train_pic, y_train = get_X_y(train_df)

X_val_pic,  y_val = get_X_y(val_df)

X_test_pic, y_test = get_X_y(test_df)









print("Initializing and Compiling model...")
y_train_one_hot = to_categorical(y_train)
y_val_one_hot = to_categorical(y_val)
y_test_one_hot = to_categorical(y_test)

#############################

activation = 'sigmoid'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (256, 256, 3)))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())

#Add layers for deep learning prediction
x = feature_extractor.output  
x = Dense(128, activation = activation, kernel_initializer = 'he_uniform')(x)
prediction_layer = Dense(1487, activation = 'softmax')(x)

# y_train.shape

# Make a new model combining both feature extractor and x
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(cnn_model.summary())

# y_train.shape

figPath = '/VisualLogs' + 

callbacks = [TrainingMonitor(figPath)]


#Train the CNN model
history = cnn_model.fit(x=X_train_pic, y=y_train, batch_size=16, epochs=1, validation_data = (X_val_pic, y_val), callbacks=callbacks)



X_for_RF = feature_extractor.predict(X_train_pic) #This is out X input to RF

#RANDOM FOREST

RF_model = RandomForestClassifier(n_estimators = 100, random_state = 42)

# Train the model on training data
RF_model.fit(X_for_RF, y_train) #For sklearn no one hot encoding
