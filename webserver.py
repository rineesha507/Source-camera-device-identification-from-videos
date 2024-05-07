import imghdr   #determine the type of image contained in a file
import tempfile  # create temporary files 

from flask import Flask, render_template, request  #handle HTTP requests and responses.
import matplotlib.pyplot as plt  # Matplotlib is a plotting library for Python. 
from PIL import Image #PIL (Python Imaging Library) is used for opening, manipulating, and saving many different image file formats
import cv2 #OpenCV is a computer vision library,
import numpy as np  #numerical computing,used for handling arrays and mathematical operations.
from keras.preprocessing.image import ImageDataGenerator  #ImageDataGenerator is used for data augmentation.
from keras.models import load_model  #load a pre-trained model from a file using Keras.
from io import BytesIO  #BytesIO class from the io modile,handle byte streams in memory.
import base64 #encoding and decoding base64 data
import os #portable way of using operating system

os.environ['CUDA_VISIBLE_DEVICES'] = '' #Environment Variable Setup:


app = Flask(__name__)  #Creating Flask App Instance


img_size = (224, 224)
batch_size = 32

dataset_dir = "split_directory/train"

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Generate training data from the dataset directory
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the trained model
model = load_model("./mobilenet_model.h5")


# Function to preprocess the new image
def preprocess_image(uploaded_image):
    img = uploaded_image
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Function to get the class label from the predicted class probabilities
def get_class_label(predictions):
    class_index = np.argmax(predictions)
    class_label = list(train_generator.class_indices.keys())[class_index]
    return class_label


@app.route('/', methods=['GET', 'POST'])  # defines a route for the root URL
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:   #checks if the uploaded file is present in the request
            return 'No file part'  
        file = request.files['file']   # retrieves the uploaded file from the request object.
        if file.filename == '':  
            return 'No selected file' 
        if file:  # checks if a file object exists
            file_bytes = BytesIO(file.read())  # reads the content of the uploaded file and stores it in a BytesIO object

            # Check if the file is an image or a video
            file_format = imghdr.what(None, h=file_bytes.read())   #imghdr module to determine the format of the uploaded file (e.g., JPEG, PNG, GIF)
            file_bytes.seek(0)

            if file_format in ['jpeg', 'png', 'gif']:
                # If the file is an image, preprocess it
                new_image = preprocess_image(Image.open(file_bytes))
            else:
                # If the file is a video, capture one frame and preprocess it
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                file.seek(0)
                file.save(temp_file.name)
                video = cv2.VideoCapture(temp_file.name)
                ret, frame = video.read()
                if not ret:
                    return 'Could not read video'
                new_image = preprocess_image(Image.fromarray(frame))
                if temp_file and temp_file.name:
                    os.remove(temp_file.name)
            predictions = model.predict(new_image)
            phone = get_class_label(predictions)
            return phone
    else:
        return '''
        <!doctype html>
        <title>Upload an Image</title>
        <h1>Source camera device identification from video</h1>
       <form method="post" enctype="multipart/form-data" action="/">
          <input type="file" name="file">
          <input type="submit" value="Upload">
        </form>
        </script>
        '''


if __name__ == '__main__': # starts the Flask application when the script is executed directly 
    app.run(debug=True)
