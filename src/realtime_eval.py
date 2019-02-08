## Code based on Pyimaagesearch tutorial
## Source: https://www.pyimagesearch.com/

from imutils.video import VideoStream, FPS
from keras.models import model_from_json
from transform import four_point_transform, order_points
from skimage.filters import threshold_local
import imutils
import numpy as np 
import argparse
import time
import cv2
import os
import tensorflow as tf 
import random




################## Training Model ##############################
# # load the data
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # reshaping data to keras API
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# load weights into new model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("[INFO] Training model loaded from disk")

# compiling the model
loaded_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



################## Video Feed ##################################
# initilize the video stream
print ("[INFO] Starting video stream...")
vs = VideoStream(src=2).start()
time.sleep(1.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:

	##### Video capture and edge detection
	# capture the frame from the video stream and resize it to 400 pixel
	frame = vs.read()
	frame2 = frame.copy()

	# ratio
	ratio = frame.shape[0] / 500.0

	# reduce the size
	frame = imutils.resize(frame, height=500)
	frame2 = imutils.resize(frame, height=500)

	# convert the image to grayscale, blur it and find edges in the image
	gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussianBlur(gray, (5,5), 0) 
	# works better without the GaussianBlur
	edged = cv2.Canny(gray, 75, 200)

	# show the original image and the edge detected image
	# print ("Step 1: Edge detection")

	# cv2.imshow("Frame", frame)
	# cv2.imshow("Edged", edged)
	

	##### Find the Contour
	# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	# loop opver the contours
	
	flag = False 		# safety flag
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then we can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			flag = True 
			break
	# if 4 contours don't exist, go to the top of the loop
	if flag == False:	# safety flag
		continue		
	

	# show the contour (outline) of the piece of paper
	# print ("Step 2: Find contours of paper")

	cv2.drawContours(frame2, [screenCnt], -11, (0, 255, 0), 2)
	# cv2.imshow("Outline", frame2)


	##### Image transform
	# apply the four point transform to obtain a top-down view of the original image
	warped = four_point_transform(frame, screenCnt.reshape(4,2) * ratio)

	# convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	resizeWarped = imutils.resize(warped, height = 32)

	T = threshold_local(warped, 101, offset = 10, method = "gaussian")
	T2 = threshold_local(resizeWarped, 9, offset = 10, method = "gaussian")

	warped = (warped > T).astype("uint8") * 255
	resizeWarped = (resizeWarped > T2).astype("uint8") * 255

	# flip the pixel intensity to match the MNIST data format
	for h in range(resizeWarped.shape[0]):
		for w in range(resizeWarped.shape[1]):
			if resizeWarped[h,w] == 0:
				resizeWarped[h,w] = 255
			elif resizeWarped[h,w] == 255:
				resizeWarped[h,w] = 0

	# crop the image
	cropped = resizeWarped[2:30, 2:30]

	# put another safety flag
	flag = False 

	if cropped.shape[0] == 28 and cropped.shape[1] == 28:
		flag = True

	if flag == False:
		continue

	# show the original and scanned image
	# print ("Step 3: Apply perspective transform")
	cv2.imshow("Warped", warped)
	# cv2.imshow("Cropped", cropped)
	# cv2.imshow("Outline", frame2)

	
	##### Image prediction
	# array format change conforming keras input format
	cropped = np.expand_dims(cropped, axis=0)
	cropped = np.expand_dims(cropped, axis=3)
	cropped = cropped.astype('float32')
	cropped /= 255	

	# predict the hand-digit number using trained model
	pred = loaded_model.predict(cropped)	
	pred_label = np.argmax(pred)
	print ("Prediction:", pred)
	
	# add label on the screen
	text_label = "{:}: {:.2f}%".format(pred_label, pred[0][pred_label] * 100)
	cv2.putText(frame2, text_label, (screenCnt[0][0][0],screenCnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cv2.imshow("Outline", frame2)


	##### Abort the video
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
	fps.update()


fps.stop()
print ("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print ("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


cv2.waitKey(0)
cv2.destroyAllWindows()
vs.stop()

"""
Copyright 2019 James Sohn

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""