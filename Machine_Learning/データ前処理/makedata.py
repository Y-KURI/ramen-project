from sklearn import cross_validation
from PIL import Image
import os, glob
import numpy as np

style_dir = "/ramen-image(2)"
styles = ["miso", "sio", "tonkotsu", "shouyu"]
nb_classes = len(styles)

image_w = 64
image_h = 64
pixels = image_w * image_h * 3

X = []
Y = []
for idx, sty in enumerate(styles):
    label = [0 for i in range(nb_classes)]
    label[idx]
    image_dir = style_dir + "/" + sty
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = \
    cross_validation.train_test_split(X, Y)
XY = (X_train, X_test, Y_train, Y_test)
np.save("/ramen_image(2)/4obj.npy", XY)

print("ok", len(Y))
