import streamlit as st
import pandas as pd
import numpy as np
import os, urllib, cv2
import tensorflow as tf
from io import BytesIO
from PIL import Image
import os
import pytesseract
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Input, Embedding, Flatten, Conv2DTranspose, concatenate, UpSampling2D,Conv2D, MaxPooling1D

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class tbl_decoder(tf.keras.layers.Layer):
  def __init__(self, name = "Table_mask"):
    super().__init__(name = name)
    self.conv1 = Conv2D(filters=512, kernel_size=(1,1), activation='relu')
    self.umsample1 = UpSampling2D(size = (2,2),)
    self.umsample2 = UpSampling2D(size = (2,2),)
    self.umsample3 = UpSampling2D(size = (2,2),)
    self.umsample4 = UpSampling2D(size = (2,2),)
    self.convtranspose = Conv2DTranspose( filters=3, kernel_size=3, strides=2, padding = 'same')

  def call(self, X):

    input,pool_3,pool_4 = X[0],X[1],X[2]
    x = self.conv1(input)
    x = self.umsample1(x)
    x = concatenate([x, pool_4])
    x = self.umsample2(x)
    x = concatenate([x, pool_3])
    x = self.umsample3(x)
    x = self.umsample4(x)
    x = self.convtranspose(x)

    return x
    
class col_decoder(tf.keras.layers.Layer):
  def __init__(self, name = "Column_mask"):
    super().__init__(name = name)
    self.conv1 = Conv2D(filters=512, kernel_size=(1,1), activation='relu')
    self.drop  = Dropout(0.8)
    self.conv2 = Conv2D(filters=512, kernel_size=(1,1), activation='relu')
    self.umsample1 = UpSampling2D(size = (2,2),)
    self.umsample2 = UpSampling2D(size = (2,2),)
    self.umsample3 = UpSampling2D(size = (2,2),)
    self.umsample4 = UpSampling2D(size = (2,2),)
    self.convtranspose = Conv2DTranspose( filters=3, kernel_size=3, strides=2, padding = 'same')

  def call(self, X):

    input,pool_3,pool_4 = X[0],X[1],X[2]
    x = self.conv1(input)
    x = self.drop(x)
    x = self.conv2(x)
    x = self.umsample1(x)
    x = concatenate([x, pool_4])
    x = self.umsample2(x)
    x = concatenate([x, pool_3])
    x = self.umsample3(x)
    x = self.umsample4(x)
    x = self.convtranspose(x)

    return x
    


@st.cache(allow_output_mutation = True)
def load_model():
  input = Input(shape=(1024,1024,3))
  vgg19 = tf.keras.applications.VGG19(include_top=False, weights = 'imagenet', input_tensor=input, classes= 1000)

  x = vgg19.output
  pool_3 = vgg19.get_layer('block3_pool').output
  pool_4 = vgg19.get_layer('block4_pool').output

  x = Conv2D(512, (1, 1), activation = 'relu', name='block6_conv1')(x)
  x = Dropout(0.8, name='block6_dropout1')(x)
  x = Conv2D(512, (1, 1), activation = 'relu', name='block6_conv2')(x)
  x = Dropout(0.8, name = 'block6_dropout2')(x)

  Table_Decoder  = tbl_decoder()
  Column_Decoder = col_decoder()

  output1 = Table_Decoder([x, pool_3, pool_4])
  output2 = Column_Decoder([x, pool_3, pool_4])

  model = Model(inputs = input, outputs= [output1,output2], name = "TableNet")
    
  model.load_weights(os.getcwd() + '\\weights-05-0.1391.hdf5')
  return model

def get_mask(mask):
  mask = tf.argmax(mask, axis=-1)
  mask = mask[..., tf.newaxis]
  return mask[0]

def table_detection(img, model) :
  """Detects and returns the table(s) in an image"""
  

  image = tf.keras.preprocessing.image.img_to_array(img)
  image = tf.image.resize(image, [1024, 1024])  #Decode a JPEG-encoded image to a uint8 tensor
  image = tf.cast(image, tf.float32) / 255.0 # normalizing image
    
  mask1, mask2 = model.predict(image[np.newaxis,:,:,:])
  table_mask, column_mask = get_mask(mask1), get_mask(mask2)
        
  img_org = tf.keras.preprocessing.image.array_to_img(image)

  img_org = img_org.resize((1024,1024),Image.ANTIALIAS)

  table_mask = tf.keras.preprocessing.image.array_to_img(table_mask)
  table_mask = table_mask.resize((1024,1024),Image.ANTIALIAS)

  column_mask = tf.keras.preprocessing.image.array_to_img(column_mask)
  column_mask = column_mask.resize((1024,1024),Image.ANTIALIAS)

  img_mask = table_mask.convert('L')

  img_org.putalpha(img_mask)
    
  st.header("Processed Table :")
  st.image(img_org, caption='Table Detected')
  return img_org, column_mask

def get_text(img_org, col_mask):
  img = cv2.cvtColor(np.asarray(img_org), cv2.COLOR_RGB2GRAY)

  thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
  img_bin = 255-img_bin

  # Length(width) of kernel as 100th of total width
  kernel_len = np.array(img).shape[1]//100
  # Defining a vertical kernel to detect all vertical lines of image 
  ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
  # Defining a horizontal kernel to detect all horizontal lines of image
  hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
  # A kernel of 2x2
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

  #Use vertical kernel to detect and save the vertical lines in a jpg
  image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
  vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

  #Use horizontal kernel to detect and save the horizontal lines in a jpg
  image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
  horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

  # Combine horizontal and vertical lines in a new third image, with both having same weight.
  img_vh = cv2.addWeighted(vertical_lines, 0.9, horizontal_lines, 0.1, 0.0 )
  #Eroding and thesholding the image
  img_vh = cv2.erode(~img_vh, kernel, iterations=2)
  thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY)
  bitxor = cv2.bitwise_xor(img,img_vh)
  bitnot = cv2.bitwise_not(bitxor)  

  im1=tf.keras.preprocessing.image.array_to_img(bitnot[:,:,np.newaxis])
  im1 = im1.resize((1024,1024),Image.ANTIALIAS)

  img_mask = col_mask
  img_mask = img_mask.convert('L')

  im1.putalpha(img_mask)


  st.write("\n")
  st.write("-"*90)
  st.write("\n")
  st.subheader("RETRIEVED TEXT :")
  #st.write("RETRIEVED TEXT :")
  st.write("\n")

  
  text_list = pytesseract.image_to_string(im1 , lang='eng' )
  text_list = text_list.split('\n')
  while("" in text_list)  :
    text_list.remove("")
  while(" " in text_list)  :
    text_list.remove(" ")
  while("  " in text_list) :
    text_list.remove("  ")

  for i in text_list:
    st.write(i)
      
def get_table(image):
    with st.spinner("Loading Model into Memory"):
        model = load_model()
    st.subheader("Checking for Tables...")
    img_org, column_mask = table_detection(image,model)
    get_text(img_org, column_mask)    


st.title("Table Detection & Data Extraction")

st.sidebar.title("Upload Image")

def getimage():
    st.subheader("Choose an Image...")
    uploaded_file = st.file_uploader("", type="bmp")
    if uploaded_file is not None:
        imag = Image.open(uploaded_file)
        st.image(imag, caption='Uploaded Image.', use_column_width=False)
        if st.sidebar.button("Detect Tables"):
            get_table(imag)    
        
        
def main():
    
    #app_mode = st.sidebar.selectbox("Choose the app mode",["Show instructions", "Run the app", "Show the source code"])
    #if st.sidebar.button("Upload Image"):
    getimage()

    #if st.sidebar.button("Detect Tables"):
        #get_table(image)
        
        
if __name__ == "__main__":
    main()       
        