from flask import Flask, render_template, url_for, request,redirect,flash
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import PIL
import pathlib
import tensorflow as tf


app = Flask(__name__)
app.config['SECRET_KEY'] = "verysecret##.."

class_names = ['A','B','C','D']

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = 'static\\uploads'
app.config['UPLOAD_FOLDER'] = dir_path

model_brain = load_model(os.path.join(dir_path,'model\\TUMOR-1.h5'))
model_tuberculosis = load_model(os.path.join(dir_path,'model\\TUMOR-1.h5'))
model_cancer = load_model(os.path.join(dir_path,'model\\TUMOR-1.h5'))

##model
def test_on_image(img_path,type):
    test_image = tf.keras.utils.load_img(img_path, target_size=(240,240))
    test_image = tf.keras.utils.img_to_array(test_image)
    #print(type(test_image))
    #print(test_image.shape)
    test_image = test_image.reshape(240, 240, 3)
    test_image = np.expand_dims(test_image, axis=0) 

    if(type=='brain'):
        result = model_brain.predict(test_image)

    if(type=='tuberculosis'):
        result = model_tuberculosis.predict(test_image)

    if(type=='cancer'):
        result = model_cancer.predict(test_image)
    #print(type(result))
    p = max(result.tolist())
    return(class_names[p.index(max(p))])


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file'] 
        type_disease=(request.form['illness']).lower()

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            file_name = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],image_path,file_name)
            file.save(file_path)
            result = test_on_image(file_path,type_disease)

    return render_template('predict.html',output =result,img_pth=file_name)



if __name__ == "__main__":
    app.run()

