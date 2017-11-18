from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, highway_conv_2d
from tflearn.layers.normalization import local_response_normalization,\
											batch_normalization
from tflearn.layers.estimator import regression

def network():
	# input
	net = input_data(shape=[None, 21, 21, 1], name='input')

	# H(2)
	for i in range(2):
		for j in [3, 2, 1]: 
			net = highway_conv_2d(net, 16, j, activation='elu')
		net = max_pool_2d(net, 2)
		net = batch_normalization(net)
	
	# 2D(32)
	net = conv_2d(net, 32, 3, activation='relu', regularizer="L2")
	net = max_pool_2d(net, 2)
	net = local_response_normalization(net)

	# F(128)
	net = fully_connected(net, 128, activation='elu')
	net = dropout(net, 0.5)

	# output
	net = fully_connected(net, 2, activation='softmax')
	return regression(net, optimizer='adam', learning_rate=0.003,
						loss='categorical_crossentropy', name='target')
