import numpy as np 
# convert_tflite_to_header.py
# Open the TensorFlow Lite model file in binary read mode
with open("addition_model.tflite", "rb") as f:
    tflite_model = f.read()  # Read the entire file into a variable

# Convert the TensorFlow Lite model to a numpy array of uint8 (unsigned 8-bit integers)
tflite_model_as_c_array = np.array(list(tflite_model), dtype=np.uint8)

# Open a new file in write mode to write the C header
with open("addition_model.h", "w") as f:
    # Write the header guard
    f.write("#ifndef ADDITION_MODEL_H\n")
    f.write("#define ADDITION_MODEL_H\n\n")

    # Begin the C array declaration
    f.write("unsigned char addition_model_tflite[] = {")
    
    # Iterate over the numpy array and write each byte to the file as a hex value
    for i, byte in enumerate(tflite_model_as_c_array):
        if i % 12 == 0:  # Add a newline every 12 bytes for readability
            f.write("\n")
        f.write(f"0x{byte:02x}, ")  # Write the byte as a hex value with a comma and space

    # Close the C array declaration
    f.write("\n};\n\n")
    
    # Write the length of the array as an unsigned int
    f.write("unsigned int addition_model_tflite_len = {};\n".format(len(tflite_model_as_c_array)))
    
    # Close the header guard
    f.write("#endif // ADDITION_MODEL_H\n")
