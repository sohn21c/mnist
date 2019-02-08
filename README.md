## Hand-written digit recognition with USB camera
#### _[James Sohn (Click to see the portfolio)](http://sohn21c.github.io)_
#### _Jan 2019_

## Objective  
To build a system to recognize hand-written digit on the paper with usb camera. Convolutional Neural Network model is to used to train the MNIST digit data set with Keras API with TensorFlow backend.  

## Video demonstration
The video below shows the demo of recognizing the hand-written number on the sticky with the usb camera.  
[![YouTube](https://github.com/sohn21c/mnist/blob/master/img/video_thumbnail.jpg?raw=true)](https://youtu.be/ipuE6w1iIO4)  

## Software
#### Convolutional Neural Network training  
`mnist_digit.ipynb`: Full blown trainig with visualization usign Keras API. Evaluation on test set achives higher than 99% accuracy. Model layout used for the training is as shown in the picture below.  
![img](https://github.com/sohn21c/mnist/blob/master/img/training_model.jpg?raw=true)   


`mnist_digit_loaded_weights.ipynb`: Simplified version of script with saved weights from the training done in the `mnist_digit.ipynb` and imported functions for visualization. One has to run the above script first to generate `.json` and `.h5` for model import.  
Training evaluation is visualized as seen below.  
![img](https://github.com/sohn21c/mnist/blob/master/img/evaluation_sample.jpg?raw=true)


`mnist_fashion.ipynb`: _(Not related to the project)_ MNIST fashion data was used for the slightly more complicated setup of CNN training for a proof of concept.  

#### Realtime USB camera hand-written digit scanner
`realtime_eval.py`: This standalone script can be run on python(3.6.8) without any argument. Trained model from the software listed above is imported to the script and video feed activates with the 1s delay. One should specify their usb port for camera source (0 for built-in webcam and 1 or 2 for external usb).  
The softwrae uses `openCV` library to convert the RGB image to GRAY, detects the edges of the paper, extracts area inside and transforms it to up-right image. Then it processes the pixel intensity and image format to match the MNIST data set for the evaluation.  
  
#### Function scripts  
`plotter.py`: Visualization tools built with `matplotlib.pyplot` and `numpy`. It plots the image and the evaluation side by side.  

`transform.py`: Image transformation tool to accept input of 4 points in contour, rearranges them in right order, performs PerspectiveTransform with OpenCV.  
  
> `scan.py`: Script that uses the functions listed in the `transform.py` to process the still image. Try `>python scan.py --help` for necessary arugments.  
![img](https://github.com/sohn21c/mnist/blob/master/img/scan_py_sample.jpg?raw=true)

  

## Hardware
Logitech 720p USB camera

## Reference/Credit
CNN training scripts are based on official Keras training material that can be found on [link](https://www.tensorflow.org/tutorials/)

`scan.py` source code is credited to [Pyimagesearch](https://www.pyimagesearch.com/)  

## License  
Lincese added to `realtime_eval.py`. 

