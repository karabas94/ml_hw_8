import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.applications.vgg16 import VGG16

"""
3 взяти кілька зображень (не більше 3-5 - собака, авто, людина, і тд) і використати pretrained модель VGG або RESNET для класифікації цих зображень.
(моделі натреновані на датасеті imagenet).
оцінити точність класифікації моделі на цих зображеннях.
"""

res_net = ResNet50(weights='imagenet')


def class_img(res_net, img_path):
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = res_net.predict(x)

    print('Predicted:', decode_predictions(preds, top=3)[0])


print('Classifying yorkshire terrier:')
class_img(res_net, 'dog.jpg')
print()

print('Classifying car/sedan:')
class_img(res_net, 'car.jpg')
print()

vgg = VGG16(weights='imagenet', include_top=True)

img_vgg = keras.utils.load_img('cat.jpg', target_size=(224, 224))
x_vgg = keras.utils.img_to_array(img_vgg)
x_vgg = np.expand_dims(x_vgg, axis=0)
x_vgg = keras.applications.vgg16.preprocess_input(x_vgg)

features = vgg.predict(x_vgg)
print('Classifying scottish straight:')
print('Predicted:', keras.applications.vgg16.decode_predictions(features, top=3)[0])
