import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image

parser = argparse.ArgumentParser(
    description='Predict the most likely flower classification from an image.')
parser.add_argument("-i","--image", 
                    help="Path to an image, i.e. './my_images/img.jpg'", 
                    default='./test_images/')
parser.add_argument("-m","--model", 
                    help="Path to a Keras model, h5 format, i.e. './my_model.h5'",
                    default='./flower_model_1586906132.h5')
parser.add_argument("-k","--top_k", 
                    help="Number of classes to report, i.e. 5", 
                    default = 5)
parser.add_argument("-c","--category_names",help="Path to JSON formatted class (flower name) list, i.e. './my_classes.json'", default='./label_map.json')

args = vars(parser.parse_args())

loaded_model = tf.keras.models.load_model(args['model'],custom_objects={'KerasLayer':hub.KerasLayer})
with open(args['category_names'], 'r') as f:
    class_names = json.load(f)
k = args['top_k']

IMG_SHAPE = 224

im = Image.open(args['image'])
image = np.asarray(im)

def process_image(img):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_SHAPE, IMG_SHAPE))
    img /= 255
    return img.numpy()  

processed_image = process_image(image)

expanded_image = np.expand_dims(processed_image, axis=0)
prediction = loaded_model.predict(expanded_image)[0]

k *= -1 # for use with argpartition
classes = np.argpartition(prediction, k)[k:] # take top k classes
probs = prediction[classes]
classes = [str(c + 1) for c in classes] # Convert to string for dict lookup
flower_names = [class_names[c] for c in classes]

print("Probable flower names for image " + args['image'] + ": ")
for i in range(args['top_k']):
    print("Flower name: " + flower_names[i] + ", probability: " + f"{probs[i]:.10f}" + ".")

