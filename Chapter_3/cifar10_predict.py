import numpy as np
#import scipy.misc

from skimage.transform import resize
from imageio import imread

from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD

model_architecture = "./Chapter_3/cifar10_architecture.json"
model_weights = "./Chapter_3/cifar10_weights.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

img_names = ["./Chapter_3/cat-standing.jpg", "./Chapter_3/dog.jpg"]
#imgs = [np.transpose(resize(imread(img_name), (32, 32)), (2, 0, 1)).astype("float32") 
#    for img_name in img_names]
imgs = [resize(imread(img_name), (32, 32)).astype("float32") for img_name in img_names]
imgs = np.array(imgs) / 255
print("imgs.shape:", imgs.shape)

optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

#predictions = model.predict_classes(imgs)
predict_x=model.predict(imgs) 
classes_x=np.argmax(predict_x,axis=1)
print("predictions:", classes_x)