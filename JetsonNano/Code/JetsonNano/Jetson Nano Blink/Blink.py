import numpy as np
import matplotlib.pyplot as plt

# Simulate capturing an image by generating random pixel data
def capture_image():
    # Generate a 100x100 image with random RGB values (values between 0 and 255)
    image_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    return image_data

# Simulate displaying the captured image
def display_image(image_data):
    # Display the image using matplotlib
    plt.imshow(image_data)
    plt.axis('off')  # Hide axis for better image viewing
    plt.show()

# Main simulation
if __name__ == "__main__":
    # Capture image (generate random image data)
    image_data = capture_image()
    
    # Display the generated image
    display_image(image_data)
