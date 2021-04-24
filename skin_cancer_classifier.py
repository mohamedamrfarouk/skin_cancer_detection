import os
import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)
import tkinter as tk
from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
import cv2

export_path_sm = "./models/model1.h5"
print(export_path_sm)

reloaded = tf.keras.models.load_model(export_path_sm)

reloaded.summary()

#0 -> melanoma
#1 -> nevus
#2 -> seborrheic_keratosis

def the_class_is(val):
    if(val==0): return "malanoma"
    elif(val==1): return "nevus"
    else: return "seborrheic_keratosis"


def get_the_predected_class(img):
    normalizedImg = np.zeros((150, 150))
    normalizedImg = cv2.normalize(img, normalizedImg, 0, 1, cv2.NORM_MINMAX)
    print(normalizedImg)

    img = np.array([normalizedImg])
    print(img.shape)
    predictions_single = reloaded.predict(img)
    print("the predected class is:", the_class_is(np.argmax(predictions_single)))
    return the_class_is(np.argmax(predictions_single))



def showimage():
    fln = filedialog.askopenfilename(initialdir="./",title="select Image File")
    img = cv2.imread(fln)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
    img = cv2.resize(img,(150,150))
    im = Image.fromarray(img).resize((350,350))
    im.thumbnail((350,350))
    my_image = ImageTk.PhotoImage(image=im)

    the_predected_clase = get_the_predected_class(img)

    # lbl2 = Label(root, text=("the_predected_class:  "+the_predected_clase))
    lbl2.config(font=(44) , text=("the_predected_class:    "+the_predected_clase))
    lbl2.pack(side=TOP, pady=15)

    lbl.configure(image=my_image)
    lbl.image = my_image
    lbl.pack(side=TOP)


root = Tk()

frm = Frame(root)
frm.pack(side=BOTTOM, padx=15, pady=15)

lbl=Label(root)
lbl.pack(side=TOP)
lbl2=Label(root)
lbl2.pack(side=TOP)
btn = Button(root, text="Browse Image", command=showimage ,height = 10, width = 20)
btn.pack(side=tk.LEFT)

btn2=Button(root, text="Exit", command=lambda: exit()  , height = 10, width = 20)
btn2.pack(side=tk.LEFT, padx=15)

root.title('image browser')
root.geometry("600x600")
root.mainloop()