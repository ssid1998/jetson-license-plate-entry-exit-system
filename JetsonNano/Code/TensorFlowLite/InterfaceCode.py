		import tensorflow as tf
		# Load TFLite model and allocate tensors.
		interpreter = tf.lite.Interpreter(model_path=
		"model.tflite")
		interpreter.allocate_tensors()
		# Get input and output tensors.
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		# Set the value of the input tensor.
		interpreter.set_tensor(input_details[0]['index'], 
		input_data)
		# Run the model.
		interpreter.invoke()
		# Get the value of the output tensor.
		output_data = interpreter.get_tensor(output_details[0]
		['index'])