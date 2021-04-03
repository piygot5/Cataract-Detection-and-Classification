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
  image_test = cv2.imread("./static/inputimages/"+file,0)
  image_test = cv2.resize(image_test, (size, size), interpolation = cv2.INTER_AREA)
  glcm_test=[]
  images_sift_test=[]
  img_arr_test = np.array(image_test)
  gCoMat = greycomatrix(img_arr_test, [1], [0],256,symmetric=True, normed=True) # Co-occurance matrix
  contrast = greycoprops(gCoMat, prop='contrast')[0][0]
  dissimilarity = greycoprops(gCoMat, prop='dissimilarity')[0][0]
  homogeneity = greycoprops(gCoMat, prop='homogeneity')[0][0]
  energy = greycoprops(gCoMat, prop='energy')[0][0]
  correlation = greycoprops(gCoMat, prop='correlation')[0][0]
  keypoints, descriptors = sift.detectAndCompute(image_test,None)
  descriptors=np.array(descriptors)
  descriptors=descriptors.flatten()
  glcm_test.append([contrast,dissimilarity,homogeneity,energy,correlation])
  glcm_test=np.array(glcm_test)
  images_sift_test.append(descriptors[:2304])
  images_sift_test=np.array(images_sift_test)
  images_sift_glcm_test=np.concatenate((images_sift_test,glcm_test),axis=1)
  result = ""
  if(log_pickle_model.predict(images_sift_glcm_test)==1):
      result = "Cataract Detected"
  else:
      result = "Cataract Not Detected"
  return redirect(url_for('results', result = result))

@app.route('/result/<result>', methods=['GET', 'POST'])
def results(result):
  return render_template('result.html', result = result)

if __name__ == "__main__":
  app.run()