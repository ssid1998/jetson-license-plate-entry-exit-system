	import tensorflow as tf
	import numpy as np
	
	# Define the input shape
	input_shape = (2,)
	
	# Create a Sequential model
	model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=input_shape),
	tf.keras.layers.Dense(1)  # Single neuron for output
	])
	
	# Compile the model
	model.compile(optimizer='sgd', loss='mse')
	
	# Generate some training data
	X_train = np.array([[1, 2], [3, 4], [5, 6]])
	y_train = np.array([[3], [7], [11]])  
	# Corresponding outputs for addition
	
	# Train the model
	model.fit(X_train, y_train, epochs=100, verbose=0)
	
	# Save the model using the recommended method for TensorFlow 2.x
	tf.keras.models.save_model(model, 'addition_model.h5')
	
	print("Model saved as 'addition_model.h5'")
			
