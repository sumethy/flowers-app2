# version 1.0

import tensorflow as tf
from tensorflow.keras.models import load_model

from flask import request
from flask import jsonify
from flask import Flask

import numpy as np
import io

from PIL import Image

app = Flask(__name__)

import os
import uuid

import redis
r = redis.Redis(host='redis-flowerapp', port=6379, db=0)
INPUT_SIZE = 299

model = load_model('flowers.h5')
print('!!! model loaded')

@app.route("/prediction", methods=['POST'])
def flowers():
    image = request.files["file"].read()
    im = Image.open(io.BytesIO(image))
    im = im.resize((INPUT_SIZE,INPUT_SIZE))

    # save the image to /data
    filename = str(uuid.uuid4())
    filename += '.jpg'
    im.save(os.path.join("/data", filename))


    if len(im.size) == 2:
        im = im.convert("RGB")

    im_arr = np.array(im)
    im_arr_scaled = im_arr / 255.0 #3D

    im_arr_scaled_expand = np.expand_dims(im_arr_scaled, axis=0) #4D

    print('!!! preprocess complete')
    probs = model.predict(im_arr_scaled_expand)
    print('!!! inference complete')
    response = {
        'prediction': {
            'daisy' : float(probs[0][0]),
            'dandelion' : float(probs[0][1]),
            'rose' : float(probs[0][2]),
            'sunflower' : float(probs[0][3]),
            'tulip' : float(probs[0][4])
        },
        "filename": filename[:-4]
    }

    answers = {
        0: "daisy",
        1: "dandelion",
        2: "rose",
        3: "sunflower",
        4: "tulip"
    }

    r.set(filename[:-4], answers[np.argmax(probs[0,:])])

    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
