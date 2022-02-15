from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf

#character classes
Classes = ['ana', 'ath', 'ba', 'bha', 'cha', 'char', 'chha', 'chhyya', 'da', 'dda', 'ddha', 'dha', 'dui', 'ek', 'ga', 'gha', 'gya', 'ha', 'ja', 'jha', 'ka', 'kha',
           'kna', 'la', 'ma', 'na', 'nau', 'pa', 'pach', 'pha', 'ra', 'sat', 'sha1', 'sha2', 'sha3', 'sunya', 'ta', 'tha', 'tin', 'tra', 'tta', 'ttha', 'wo', 'xa', 'ya', 'yan']

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static','images')

app.config['IMAGE_FOLDER']=UPLOAD_FOLDER

@app.route('/', methods=['GET'])

def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])

def predict():
    IMG_SIZE = 32
    imageFile=request.files['imageFile']
    image_path = os.path.join(app.config['IMAGE_FOLDER'], imageFile.filename )
    imageFile.save(image_path)
    #loading model
    model1 = tf.keras.models.load_model('../nhwcr.hdf5')
    #prediction on our image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE),interpolation = cv2.INTER_AREA)
    img = tf.keras.utils.normalize(img,axis=1)
    img1 = np.array(img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    pred = Classes[np.argmax(model1.predict(img1))]


    return render_template('index.html', character=pred, imagepath=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)