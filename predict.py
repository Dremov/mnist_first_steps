# Modify 'test1.jpg' and 'test2.jpg' to the images you want to predict on

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images
img_width, img_height = 28, 28

# load the model we saved
model = load_model('save/mnist_model.h5')
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# predicting images
img = image.load_img('input9.png', grayscale=True, target_size=(img_width, img_height))
img = np.asarray(img)
img = img.astype('float32')
img /= 255
img = img.reshape(1, 1, 28, 28)

classMatrix = model.predict(img, batch_size=128)
classes = model.predict_classes(img, batch_size=128)

print(classMatrix)
print(classes)
