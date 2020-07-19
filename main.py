from flask import Flask, request
from flask_restful import Resource, Api

from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

import requests as req
from io import BytesIO


# Load the model
model = load_model('keras_model.h5')

def get_img(url):
    response = req.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def img_preprocessing(url):
    image = get_img(url)
    image = ImageOps.fit(image, (224,224), Image.ANTIALIAS)
    image = np.asarray(image)
    image = (image.astype(np.float32) / 127.0) - 1
    image = np.array([image])
    return(image)


def predict(url):
    img = img_preprocessing(url)
    return(np.argmax(model.predict(img)))

app = Flask(__name__)
api = Api(app)


class GetPrediction(Resource):
    def get(self, filename):
        url = 'https://binary-cdk.herokuapp.com/static/uploads/' + filename
        label = predict(url)
        return int(label)
    
api.add_resource(GetPrediction, '/<string:filename>')


if __name__ == '__main__':
    app.run()