import os    #os for system operations,
import cv2    #cv2 for OpenCV (computer vision library), 
import numpy as np   #numpy for numerical operations, 
import pandas as pd   #pandas for data manipulation, 
from keras.preprocessing.image import ImageDataGenerator    #ImageDataGenerator from Keras for data augmentation,
from keras.models import load_model     #load_model from Keras for loading the trained model,
import tempfile    #tempfile for temporary file handling

# Define your dataset directory
dataset_dir = "split_directory"

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(     # specifies various transformations to be applied to the images during training.
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
train_generator = datagen.flow_from_directory(   #flow_from_directory() method is used  generates data batches for training the model.
    f"{dataset_dir}/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Now you can print the class indices
#print("Training Class Indices:", train_generator.class_indices)

# Load the trained model
model = load_model("./mobilenet_model.h5")


# Function to preprocess the new image
def preprocess_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to get the class label from the predicted class probabilities
def get_class_label(predictions):
    class_index = np.argmax(predictions)
    class_label = list(train_generator.class_indices.keys())[class_index]
    return class_label


# Path to the new image file
# new_image_path = "./split_directory/test/Lenovo/frame_1_7.jpg"
#new_image_path = "./split_directory/test/samsung/frame_2_10.jpg"
#new_image_path = "./split_directory/test/apple_ipadMini/frame_3_16.jpg"

DOCUMENT_DIR = "/mnt/c/Users/rinee/Documents/video_capture_prediction/train2"
video = cv2.VideoCapture(f'{DOCUMENT_DIR}/2nd apple4/D09_V_flatYT_still_0002.mp4')
#video = cv2.VideoCapture(f'{DOCUMENT_DIR}/microsoft_lumia/D17_V_indoorYT_move_0001.mp4')
#video = cv2.VideoCapture(f'{DOCUMENT_DIR}/samsung/D01_V_outdoor_panrot_0001.mp4')
#video = cv2.VideoCapture(f'{DOCUMENT_DIR}/real data sony testing/D12_V_flatWA_move_0003.mp4')


#checks if the video capture object video is valid.
if video is not None:
    ret, frame = video.read()   #video.read() reads the next frame from the video capture object,
    if not ret:      #ret indicates whether the operation was successful.
        print('Could not read video')
        exit()
    temp_file = tempfile.NamedTemporaryFile(delete=False)   #creates a temporary file to store the newly captured frame.tempfile.NamedTemporaryFile() creates a temporary file object.
    new_image_path = '{temp_file.name}.png'   #temp_file.name retrieves the name of the temporary file.
    cv2.imwrite(new_image_path, frame)  #cv2.imwrite() writes the captured frame to the temporary file.

#Preprocess the new image
new_image = preprocess_image(new_image_path)

# Use the model to predict the class
predictions = model.predict(new_image)
# print(predictions)

# Get the predicted class label
predicted_class = get_class_label(predictions)
print("Predicted Class:", predicted_class)
#os.remove(temp_file.name)

# print prediction table

# Assuming train_generator is your training data generator
class_indices = train_generator.class_indices
# Get the class labels (device labels) from the train generator
class_labels = list(class_indices.keys())

# Create a DataFrame to store the predictions
df = pd.DataFrame(columns=['Class', 'Prediction'])

# For each prediction, get the corresponding class label and add it to the DataFrame
for i, prediction in enumerate(predictions):
    predicted_class_index = np.argmax(prediction)  #np.argmax(prediction) finds the index with the highest prediction value.
    predicted_class_label = class_labels[predicted_class_index]
    print(predicted_class_label)
    df.loc[i] = [predicted_class_label, prediction[predicted_class_index]]  #df.loc[i] adds the predicted class label and prediction value to the DataFrame.


# Display the DataFrame
print(df)

