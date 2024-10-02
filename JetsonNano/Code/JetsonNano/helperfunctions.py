# define helper functions
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline
  """%matplotlib inline, represents the magic function %matplotlib,
  which specifies the backend for matplotlib, and with the argument 
  inline you can display the graph and make the plot interactive."""
  image = cv2.imread(path)
  height, width = image.shape[:2]
  """img.shape returns (Height, Width, Number of Channels) 
  where Height represents the number of pixel rows in the image or the number of pixels in each column of the image array."""
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
  #where 3*width and 3*height scale factors along x and y
  #the interpolation flag refers to which method we are going to use
  fig = plt.gcf()
  #plt.gcf () allows you to get a reference to the current figure when using pyplot.
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

# use this to upload files to the cloud Virtual Machine
def upload():
  from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)

# use this to download a file from cloud VM
def download(path):
  from google.colab import files
  files.download(path)