import matplotlib.pyplot as plt
import numpy as np

def plot_image(i, prediction_array, true_label, img):
    prediction_array, true_label, img = prediction_array[i], true_label[i], img[i]
    
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img.reshape(28,28), cmap="Greys") # reshape the image data back to 28x28
    
    predicted = np.argmax(prediction_array)
    
    if predicted == true_label:
        color = 'g'
    else:
        color = 'r'
        
    plt.xlabel("{} {:2.0f}% ({})".format(predicted, 100*np.max(prediction_array), true_label), color=color)
    

def plot_value_array(i, prediction_array, true_label):
    
    prediction_array, true_label = prediction_array[i], true_label[i]
    
    plt.grid(False)
    plt.xticks(range(10), range(10))
    plt.yticks([])
    barplot = plt.bar(range(10), prediction_array, color="#777777")
    plt.ylim([0,1])
    predicted = np.argmax(prediction_array)
    
    barplot[predicted].set_color('red')
    barplot[true_label].set_color('green')

