from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import *

# input
model = Sequential()
model.add(Dense(441, input_shape=(21,21,1)))

# H(2)
for i in range(2):
	for j in [3, 2, 1]:
		model.add(Conv2D(16, j, activation='elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())

# F(128)
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())

# output
model.add(Dense(2, activation='softmax'))
model.compile(RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
