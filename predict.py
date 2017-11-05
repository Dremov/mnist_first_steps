# Modify 'test1.jpg' and 'test2.jpg' to the images you want to predict on

from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')

# dimensions of our images
img_width, img_height = 28, 28

# load the model we saved
model = load_model('save/mnist_model.h5')

# predicting images
img = image.load_img('data/test_numbers/input9.png', grayscale=True, target_size=(img_width, img_height))
img = np.asarray(img)

img = img.astype('float32')
img /= 255
img = img.reshape(1, 1, 28, 28)

classMatrix = model.predict(img, batch_size=128)
classes = model.predict_classes(img, batch_size=128)

print(classMatrix)
print(classes)
