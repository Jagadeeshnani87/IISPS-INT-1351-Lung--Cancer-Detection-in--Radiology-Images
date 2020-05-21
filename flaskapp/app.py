from __future__ import division, print_function
import base64
import numpy as np
import io
import os
from werkzeug.utils import secure_filename
from PIL import Image
import keras
import tensorflow as tf
from keras import backend as k
from keras.models import load_model
from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import jsonify
from flask import request
from  flask import render_template
from flask import Flask
global graph

graph = tf.get_default_graph()
app = Flask(__name__)
def get_model():
    global models
    model = load_model('rnn.h5')
    print("*Model loaded!")

print(" * Loading keras model...")
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')

@app.route("/predict",methods=["POST"])
def upload():
   
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            try:
                pred = get_model.predict_classes(x)
                p=get_model.predict(x)
                if(pred==0):
                    text="normal"
                else:
                    text="abnormal"
    retrun text
    
                               
            except AttributeError:
                print("prediction:")
    
    

if __name__ == '__main__':
    app.run(debug=True,threaded = False)

    
        
