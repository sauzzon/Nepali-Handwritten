from flask import Flask, render_template, request
import base64
from PIL import Image
from io import BytesIO
import os
import cv2
from itsdangerous import base64_decode
import numpy as np
import tensorflow as tf

#character classes
Classes = ['ण', '८', 'ब', 'भ', 'च', 
'४', 'छ', 'क्ष','ड', 'द', 'ध','ढ', 
 '२', '१', 'ग', 'घ', 'ज्ञ', 'ह', 
'ज', 'झ', 'क', 'ख',
'ङ', 'ल', 'म', 'न', '९', 'प', '५', 
'फ', 'र', '७', 'श', 'ष', 'स', 
'o', 'ट', 'ठ', '३', 'त्र', 'त', 
'थ', 'व', '६', 'य', 'ञ']

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static','images')

app.config['IMAGE_FOLDER']=UPLOAD_FOLDER

@app.route('/', methods=['GET'])

def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])

def predict():
    IMG_SIZE = 32
    if not request.files.get('imageFile'):
        imageFileEncoded=request.values.get('imageFileEncoded')
        if not imageFileEncoded:
            return render_template('index.html',error="No File Choosen")
        starter=imageFileEncoded.find(',')
        image_data = imageFileEncoded[starter+1:]
        image_data = bytes(image_data, encoding="ascii")
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        
        filename = 'canvasimage.png'  # I assume you have a way of picking unique filenames

        image_path = os.path.join(app.config['IMAGE_FOLDER'], filename )
        img.save(image_path, 'png')
        
        if not imageFileEncoded:
            return render_template('index.html',error="No File Choosen")
    else:
        imageFile=request.files.get('imageFile')
        if not imageFile:
            return render_template('index.html',error="No File Choosen")
        image_path = os.path.join(app.config['IMAGE_FOLDER'], imageFile.filename )
        #save the choosen file to the server compulsory coz cv2 is reading from this path
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