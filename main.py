#tootsie roll

from flask import Flask, render_template, request, jsonify
import cv2
from PIL import Image
import numpy as np
from tensorflow import keras
from keras.models import load_model
import matplotlib
from matplotlib import pyplot as plt
from efficientnet.tfkeras import EfficientNetB7
from io import BytesIO
import base64
import urllib

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    images = [
        {'src': 'dyed-lifted-polyps.jpg', 'label': 'Dyed Lifted Polyps'},
        {'src': 'dyed-resection-margins.jpg', 'label': 'Dyed Resection Margins'},
        {'src': 'esophagitis.jpg', 'label': 'Esophagitis'},
        {'src': 'polyps.jpg', 'label': 'Polpys'},
        {'src': 'normal-cecum.jpg', 'label': 'Normal Cecum'},
        {'src': 'normal-pylorus.jpg', 'label': 'Normal Pylorus'},
        {'src': 'normal-z-line.jpg', 'label': 'Normal Z-Line'},
        {'src': 'ulcerative-colitis.jpg', 'label': 'Ulcerative Colitis'},
    ]
    return render_template('upload.html', images=images)

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image = Image.open(image)
    image = image.resize((160, 128))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image[0, :, :] = cv2.equalizeHist(image[0, :, :])
    image[:, 0, :] = cv2.equalizeHist(image[:, 0, :])
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    image = Image.fromarray(image, mode='RGB')
    image = image.quantize(kmeans=4)
    image = image.convert('RGB')
    image = np.array(image).astype('float32')/255.0
    image = image.reshape(-1, 128, 160, 3)

    model = keras.models.load_model('/home/HeavySilver/Flaskapp691/endoscopy_modelv5.h5', compile=False)
    pred = np.argmax(model.predict(image))

    class_names = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
               'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']

    image = image[0,:,:,:]
    plt.imshow(image, aspect='equal')
    plt.title(class_names[pred])
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    plt.close()

    return render_template('prediction.html', uri=uri)

if __name__ == '__main__':
    app.run(debug=True)