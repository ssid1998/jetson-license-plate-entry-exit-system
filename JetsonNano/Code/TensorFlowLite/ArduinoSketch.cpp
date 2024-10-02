			#include <TensorFlowLite.h>
			#include <Arduino_TensorFlowLite.h>
			
			// Include the TensorFlow Lite model
			#include "simple_addition_model.h" // Assuming this is your converted model header file
			
			void setup() {
				// Initialize serial communication
				Serial.begin(9600);
				
				// Initialize TensorFlow Lite interpreter
				if (!TfLite.begin(model_data, model_data_size)) {
					Serial.println("Failed to initialize TensorFlow Lite!");
					while (1);
				}
			}

			void loop() {
				// Example inference
				float input1 = 5.0;
				float input2 = 3.0;
				
				// Prepare input tensor
				TfLiteTensor* input = TfLite.getInputTensor(0);
				input->data.f[0] = input1;
				input->data.f[1] = input2;
				
				// Run inference
				TfLite.run();
				
				// Get output tensor
				TfLiteTensor* output = TfLite.getOutputTensor(0);
				float result = output->data.f[0];

				// Print result
				Serial.print(input1);
				Serial.print(" + ");
				Serial.print(input2);
				Serial.print(" = ");
				Serial.println(result);
				
				// Wait
				delay(1000);
			}