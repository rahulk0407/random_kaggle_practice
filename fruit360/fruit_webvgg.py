# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:22:40 2019

@author: rahul
"""


import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import cv2
import keras
import numpy as np
from keras.models import load_model
from keras import backend as K

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/"+filename)
			#color_result = getDominantColor(image)
			result = fruits360(image)
			redirect(url_for('upload_file',filename=filename))
			return '''
			<!doctype html>
			<title>Results</title>
			<h1>Image contains a - '''+result+'''</h1>
			<form method=post enctype=multipart/form-data>
			  <input type=file name=file>
			  <input type=submit value=Upload>
			</form>
			'''
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form method=post enctype=multipart/form-data>
	  <input type=file name=file>
	  <input type=submit value=Upload>
	</form>
	'''
    
classes=['Apple Braeburn',
         'Apple Golden 1',
         'Apricot',
         'Avocado',
         'banana',
         'Cactus fruit',
         'Cherry 1',
         'Grape Blue',
         'Guava',
         'Kiwi',
         'Lemon',
         'Mango',
         'Mango Red',
         'Onion Red',
         'Orange',
         'Papaya',
         'Peach',
         'Pineapple',
         'Pomegranate',
         'Strawberry'
         ]


def fruits360(image):
    from keras.utils import CustomObjectScope
    from keras.initializers import glorot_uniform
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        classifier = load_model('./fruit360vgg.h5')  
    image=cv2.imread("test-multiple_fruits/mango.jpg")
    image = cv2.resize(image, (100,100), interpolation = cv2.INTER_AREA)
    image = image.reshape(1,100,100,3) 
    res = classifier.predict(image, 1, verbose = 0)

    max=res[0][0]
    
    for i in range(0,len(res[0])):
        if res[0][i] > max:
            max=res[0][i]
            c=i
            x=c
    res=classes[x]
    return res
    
if __name__ == "__main__":
	app.run(host= '0.0.0.0', port=80)