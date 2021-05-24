import sys
import os
import pickle
import cv2
from skimage.feature import  greycomatrix, greycoprops
from sklearn import preprocessing
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, flash, request, redirect, url_for
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import time
import h5py
from keras.models import load_model


app = Flask(__name__)
app.debug = True


log_pickle_model = pickle.load(open("log_model.sav", 'rb'))
sift = cv2.xfeatures2d.SIFT_create()

size=128
intensity_model = load_model('squeezenet.h5')


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


@app.route('/prediction/<file>', methods=['GET', 'POST'])
def prediction(file):
  result1 = ""
  result2 = ""
  start = time.time()
  image = cv2.imread("./static/inputimages/"+file,0)
  image_test = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
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
  if(log_pickle_model.predict(images_sift_glcm_test)==1):
      result1 = "Cataract detected"
      filename = "./static/inputimages/"+file
      img = cv2.imread(filename)
      img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
      frame = img
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      sensitivity = 156
      lower_white = np.array([0,0,255-sensitivity])
      upper_white = np.array([255,sensitivity,255])
      # Threshold the HSV image to get only white colors
      mask = cv2.inRange(hsv, lower_white, upper_white)
      # Bitwise-AND mask and original image
      res = cv2.bitwise_and(frame,frame, mask= mask)
      ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
      circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.5, 100000,param1=80,param2=40,minRadius=0,maxRadius=0)
      x,y,r = 0,0,0
      if circles is not None:
          circles = np.uint16(np.around(circles))
          x,y,r = circles[0][0]
          x=int(x)
          y=int(y)
          r=int(r)
      mask = np.zeros((224,224), np.uint8)
      cv2.circle(mask,(x,y),r,(255,255,255),-1)
      masked_data = cv2.bitwise_and(frame, frame, mask=mask)
      _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
      cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
      x,y,w,h = cv2.boundingRect(cnt[0])

      # Crop masked_data
      crop = masked_data[y:y+h,x:x+w]
      crop = cv2.resize(crop, (224,224), interpolation = cv2.INTER_AREA)
      #preprocess the image
      my_image = preprocess_input(crop)
      crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
      my_image = img_to_array(crop)
      my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
      ans = intensity_model.predict(my_image)
      ans_class = np.argmax(ans)
      classes = ["Mild Cataract","Normal Cataract","Severe Cataract"]
      result2 = classes[ans_class]
  else:
      result1 = "No catarcat"
      result2 = "Normal eye"
  return redirect(url_for('results', result1 = result1, result2 = result2))

@app.route('/result/<result1>/<result2>', methods=['GET', 'POST'])
def results(result1, result2):
  return render_template('result.html', result1 = result1, result2 = result2)

if __name__ == "__main__":
  app.run()