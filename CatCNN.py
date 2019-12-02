#import Statements
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from IPython.display import SVG
from PIL import Image
from random import shuffle, choice
import numpy as np
import matplotlib.pyplot as plt
import os


#Standard image Size
Image_Size = 256
img_dir = '/Users/ahmed/Downloads/data/test_set'


# A function to label the images either as Cats or Non Cats using one-hot bbinary assignment 
def label_img(name):
	if name == "cats": 
		return np.array([1,0])


	else: 
		return np.array([0,1])


# A function that loads the Cat images, conevrts it into *bbit Gray scale and resizes it to a Standard 256 size
def load_data():
	print("Loading images...")

	train_data = []
	directories = next(os.walk(img_dir))[1]

	for dirname in directories:
		print("Loading". format(dirname))
		file_names = next(os.walk(os.path.join(img_dir, dirname)))[2]
# We load only 200 images out of the total 5000 images since we have a slow architecture 
		for i in range(200):
			image_name = choice(file_names) # Choosing a random element from the dataset
			image_path =os.path.join(img_dir, dirname, image_name)
			label = label_img(dirname)
			if "DS_Store" not in image_path:
				img = Image.open(image_path)
				img = img.convert('L')
				img = img.resize((256,256), Image.ANTIALIAS)
				train_data.append([np.array(img), label])


	return train_data


# We create a model with 6 layers (each layer contains CONV Layer/ filer , Pooling Layer & Batch Normalization) then we 
# we then add a DropOut filter to avoid OverFitting
def create_model():
  model = Sequential()
  model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', 
                   input_shape=(256, 256, 1)))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(2, activation = 'softmax'))

  return model

# Main Program to load the data and train the model using model.fit() in 10 iterations
training_data = load_data()
training_images = np.array([i[0] for i in training_data]).reshape(-1, 256,256,1)
training_labels = np.array([i[1] for i in training_data])

print("Creating_Model")
model = create_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('training The Model')
history = model.fit(training_images,training_labels, validation_split=0.25, batch_size = 200, epochs = 10, verbose = 2)
model.save("Cats.h5")






# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

