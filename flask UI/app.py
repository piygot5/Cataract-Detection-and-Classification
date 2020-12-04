import os
import pickle
import cv2
from skimage.feature import  greycomatrix, greycoprops
from sklearn import preprocessing
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, flash, request, redirect, url_for

app = Flask(__name__)
app.debug = True


#IMPORTING THE PICKLE MODEL

model = pickle.load(open("mainmodel.pkl", 'rb'))
sift = cv2.xfeatures2d.SIFT_create()
min_max_scaler = preprocessing.MinMaxScaler()
size=128

@app.route("/")
def hello():
  return render_template('./home.html')

@app.route("/about/")
def about():
  return render_template('./about.html')

@app.route("/main/")
def main():
  return render_template('./main.html')

@app.route('/upload_file/', methods = ['POST'])
def upload_file():
    UPLOAD_FOLDER = './static/inputimages'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('main.html', user_image = filename)


@app.route('/predict/<file>', methods=['GET', 'POST'])
def predict(file):
  glcm=[]
  images_sift=[]
  images_sift_glcm=[]
  image = cv2.imread("./static/inputimages/"+file,0)
  image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
  img_arr = np.array(image)
  gCoMat = greycomatrix(img_arr, [1], [0],256,symmetric=True, normed=True) # Co-occurance matrix
  contrast = greycoprops(gCoMat, prop='contrast')[0][0]
  dissimilarity = greycoprops(gCoMat, prop='dissimilarity')[0][0]
  homogeneity = greycoprops(gCoMat, prop='homogeneity')[0][0]
  energy = greycoprops(gCoMat, prop='energy')[0][0]
  correlation = greycoprops(gCoMat, prop='correlation')[0][0]
  keypoints, descriptors = sift.detectAndCompute(image,None)
  glcm.append([contrast,dissimilarity,homogeneity,energy,correlation])
  images_sift=descriptors[:18]
  images_sift=np.array(images_sift)
  images_sift=images_sift.reshape([1,2304])
  glcm=np.array(glcm)
  images_sift_glcm=np.concatenate((images_sift,glcm),axis=1)
  x_scaled = min_max_scaler.fit_transform(images_sift_glcm)
  result = ""
  value=model.predict(x_scaled)
  if(value==1):
    result = "Cataract Detected"
  else:
    result = "Cataract Not Detected"
  return redirect(url_for('results', result = result))

@app.route('/result/<result>', methods=['GET', 'POST'])
def results(result):
  return render_template('result.html', result = result)

if __name__ == "__main__":
  app.run()