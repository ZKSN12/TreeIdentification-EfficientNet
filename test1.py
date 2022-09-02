import pickle
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from  tensorflow.keras.models import load_model
import time
import os


norm_size=224
imagelist=[]
emotion_labels = {
    0: 'walnut',
    1: 'blackcherry',
    2: 'redoak',
    3: 'whitecherry',
}
emotion_classifier=load_model("best_model.hdf5")
lb = pickle.loads(open("label_bin.pickle", "rb").read())
t1=time.time()
predict_dir = 'data/test'
test11 = os.listdir(predict_dir)
for file in test11:
    filepath=os.path.join(predict_dir,file)
    image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    image = cv2.resize(image, (norm_size, norm_size), interpolation=cv2.INTER_LANCZOS4)
    image = img_to_array(image)
    imagelist.append(image)
imageList = np.array(imagelist, dtype="float") / 255.0
out = emotion_classifier.predict(imageList)
print(out)
pre = [np.argmax(i) for i in out]

class_name_list=[lb.classes_[i] for i in pre]
print(class_name_list)
t2 = time.time()
t3 = t2 - t1
print(t3)


