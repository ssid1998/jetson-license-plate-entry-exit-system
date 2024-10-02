import os
image_files = []
os.chdir(os.path.join("data", "test"))
for filename in os.listdir(os.getcwd()):
""" os.listdir() method in python is used to get the list of all files and directories in the specified directory. 
If we don't specify any directory, 
then list of files and directories in the current working directory will be returned."""
    if filename.endswith(".jpg"):
        image_files.append("data/test/" + filename)
os.chdir("..")
"""Python method chdir() changes the current working directory to the given path.It returns None in all the cases."""
with open("test.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        #Out-File uses the FilePath parameter and creates a file in the current directory
        outfile.write("\n")
    outfile.close()
os.chdir("..")