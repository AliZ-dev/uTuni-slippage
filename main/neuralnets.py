# import the necessary packages
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class network():

	def cnn_layers(self, IMG_HEIGHT, IMG_WIDTH):
		# load the VGG16 network, ensuring the head FC layer sets are left
		# off
		model = VGG16(weights="imagenet", include_top=False,
			input_tensor=Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

		return model

	def dense_layers(self, DATA_DIM, HIDDEN_TOPOLOGY):

		
		inputX = Input(shape=DATA_DIM, dtype='float32', name='numerical_input')
		for (i, layerSize) in enumerate(HIDDEN_TOPOLOGY):
			
			if i == 0:
				x = inputX

			x = Dense(layerSize, activation='relu')(x)
			#x = Dropout(0.2)(x)
			#x = LeakyReLU(alpha=0.04)(x)

		model = Model(inputs=inputX, outputs=x)
		return model

