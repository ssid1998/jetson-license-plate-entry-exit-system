# Bilder zuschneiden und neu sortieren

import glob
from os import path
from PIL import Image
import math

output_width  = 30
output_height = 80

files = glob.glob('./bad_good/*.bmp')

for i in range (len( files ) ) :
    im = Image.open( files [ i ])
    im = im.transpose( Image.ROTATE_270 )
    im = im.resize((2*output_width, output_height ) )

    imleft = im.crop ((0 , 0 , output_width , output_height ) )
    imleft.save('../training_data/bad/left_'+path.basename( files [ i ]) )
    imright = im.crop (( output_width , 0 , 2*output_width , output_height ) )
    imright = imright.transpose ( Image.FLIP_LEFT_RIGHT )
    imright.save('../training_data/good/right_'+path.basename ( files [ i ]) )