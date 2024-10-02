DATADIR = "./Datenbasis/"
CATEGORIES = [ "good" , "bad" ]
IMG_HEIGHT = 80
IMG_WIDTH = 30

training_data = [ ]
def create_training_data () :

for category in CATEGORIES :
    path = os.path.join(DATADIR , category )
    class_num = CATEGORIES.index ( category )
    for img in tqdm (os.listdir (path) ) :
        try:
            img_array = cv2.imread(os.path.join(path , img) , cv2.IMREAD_GRAYSCALE )
            new_array = cv2.resize (img_array , (IMG_WIDTH , IMG_HEIGHT ) )
            training_data.append ([ new_array , class_num ])
        except Exception as e:
           pass

 create_training_data ()