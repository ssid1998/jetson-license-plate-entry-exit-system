		import tensorflow as tf
		from tensorflow.keras.losses import MeanSquaredError
		
		# Load the Keras model with custom objects
		model = tf.keras.models.load_model('addition_model.h5', custom_objects={'mse': MeanSquaredError()})
		
		# Create a concrete function from the Keras model
		@tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
		def model_concrete_func(x):
		return model(x)
		
		# Convert the concrete fn to a TF Lite model
		converter = tf.lite.TFLiteConverter.from_concrete_functions([model_concrete_func.get_concrete_function()])
		tflite_model = converter.convert()
		
		# Save the TensorFlow Lite model to a file
		with open('addition_model.tflite', 'wb') as f:
		f.write(tflite_model)
