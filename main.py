import os 
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.models import Model

DATA_DIR = './data'
TRAIN_DIR = os.path.join(DATA_DIR, "train")
CLASS_LIST = ["cat", "dog"]
EXTENSION = ".jpg"
SPLIT_INDEX = 0.8
EPOCHS = 40
BATCH_SIZE = 32
INPUT_SHAPE = [256, 256, 3]
LEARNING_RATE = 0.00001

def get_per_class_image_list(image_list, class_name, split_iden=".", split_iden_index=0, shuffle=True):
	class_name_image_list = [os.path.join(TRAIN_DIR, image_file) for image_file in image_list if image_file.split(split_iden)[split_iden_index] == class_name]
	if shuffle:
		random.shuffle(class_name_image_list)
	print("For Class {} Found {} Images".format(class_name, len(class_name_image_list)))
	return class_name_image_list

def split_data(image_list, class_list, split_index):
	train_data_dict = {"images":[], "labels":[]}
	val_data_dict = {"images":[], "labels":[]}
	for i, class_name in enumerate(class_list):
		class_image_list = get_per_class_image_list(image_list=image_list, class_name=class_name)
		
		train_image_list = class_image_list[:int(len(class_image_list)*split_index)]
		train_label_list = [i for k in train_image_list]
		
		val_image_list = class_image_list[int(len(class_image_list)*split_index):]
		val_label_list = [i for k in val_image_list]
		
		train_data_dict["images"].extend(train_image_list)
		train_data_dict["labels"].extend(train_label_list)
		val_data_dict["images"].extend(val_image_list)
		val_data_dict["labels"].extend(val_label_list)
	return train_data_dict, val_data_dict

def get_img_file(img_path, input_shape):
	image = tf.io.read_file(img_path)
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.resize(image, [input_shape[0], input_shape[1]], antialias=True)
	image = tf.cast(image, tf.float32)/255.0
	return image

def parse_function(ip_dict, input_shape):
	label = ip_dict["labels"]
	image = get_img_file(img_path=ip_dict["images"], input_shape=input_shape)
	return image, label

def get_data_pipeline(data_dict, batch_size, input_shape):
	total_images = len(data_dict["images"])
	with tf.device('/cpu:0'):
		dataset = tf.data.Dataset.from_tensor_slices(data_dict)
		dataset = dataset.shuffle(total_images)
		dataset = dataset.map(lambda ip_dict: parse_function(ip_dict, input_shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		dataset = dataset.batch(batch_size)
		dataset = dataset.prefetch(buffer_size=1)
	return dataset

def show_batch(image_batch, label_batch, plt_title, rows=2, cols=2):
	num_images_to_show = rows * cols
	plt.figure(figsize=(8,8))
	for n in range(num_images_to_show):
		ax = plt.subplot(rows, cols, n+1)
		plt.imshow(image_batch[n], cmap='gray')
		plt.title(str(label_batch[n].numpy()))
		plt.axis('off')
	plt.suptitle(plt_title, fontsize=14)
	plt.show()


all_image_list = [img_file_name for img_file_name in os.listdir(TRAIN_DIR) if os.path.splitext(img_file_name)[-1]==EXTENSION]
train_data_dict, val_data_dict = split_data(image_list=all_image_list, class_list=CLASS_LIST, split_index=SPLIT_INDEX)
print("-"*15, "Data Sumamry", "-"*15)
print("Total Images : {}".format(len(all_image_list)))
print("Total Train Images : {} and Labels : {}".format(len(train_data_dict["images"]), len(train_data_dict["labels"])))
print("Total Val Images : {} and Labels : {}".format(len(val_data_dict["images"]), len(val_data_dict["labels"])))

train_data_pipeline = get_data_pipeline(data_dict=train_data_dict, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE)
val_data_pipeline = get_data_pipeline(data_dict=val_data_dict, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE)

'''
image_batch, label_batch = next(iter(train_data_pipeline))
print("Image shape: ", image_batch.numpy().shape)
print("Label: ", label_batch.numpy())
show_batch(image_batch, label_batch, plt_title=CLASS_LIST, rows=4, cols=4)
'''

def get_model_arch(input_shape, last_layer_activation='sigmoid'):
	input_img = Input(input_shape, name='input')
	x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='fe0_conv1')(input_img)
	x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='fe0_conv2')(x)
	x = MaxPooling2D((2, 2), padding='same', name='fe0_mp')(x)

	x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='fe1_conv1')(x)
	x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='fe1_conv2')(x)
	x = MaxPooling2D((2, 2), padding='same', name='fe1_mp')(x)
	
	x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='fe2_conv1')(x)
	x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='fe2_conv2')(x)
	x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='fe2_conv3')(x)
	x = MaxPooling2D((2, 2), padding='same', name='fe2_mp')(x)

	x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='fe3_conv1')(x)
	x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='fe3_conv2')(x)
	x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='fe3_conv3')(x)
	x = MaxPooling2D((2, 2), padding='same', name='fe3_mp')(x)

	x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='fe4_conv1')(x)
	x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='fe4_conv2')(x)
	x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='fe4_conv3')(x)
	x = MaxPooling2D((2, 2), padding='same', name='fe4_mp')(x)
	
	x = Flatten(name='feature')(x)
	x = Dense(100, activation='relu', name='fc0')(x)
	x = Dense(100, activation='relu', name='fc1')(x)
	logits = Dense(1, name='logits')(x)
	probabilities = Activation(last_layer_activation)(logits)
	model_arch = Model(inputs=input_img, outputs=probabilities)
	return model_arch

loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr = LEARNING_RATE)
metric = tf.keras.metrics.BinaryAccuracy(name="baccuracy")

model = get_model_arch(input_shape=INPUT_SHAPE)
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
print(model.summary())

history = model.fit(train_data_pipeline, epochs=EPOCHS, validation_data=val_data_pipeline, shuffle=True, verbose=1)
model.save("Weights.h5")

def plot_metric_curve(history, metric, title):
	plt.plot(history.history[metric])
	plt.plot(history.history['val_'+metric])
	plt.title(title)
	plt.ylabel(metric)
	plt.xlabel('Epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()

plot_metric_curve(history, metric="loss", title="Loss Comparison")
plot_metric_curve(history, metric="baccuracy", title="Binary Accuracy Comparison")