import tensorflow as tf   #TensorFlow is an open-source machine learning framework
from tensorflow.keras.applications import MobileNet  #Keras is an API that provides a high-level interface for building and training neural networks
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  #Dense is a type of layer in a neural network.GlobalAveragePooling2D layer is used to reduce the spatial dimensions of a tensor by taking the average value of each feature map
from tensorflow.keras.models import Model #Model is a class in Keras used for constructing neural network models.
from tensorflow.keras.preprocessing.image import ImageDataGenerator #ImageDataGenerator is a utility in Keras used for data augmentation and preprocessing of image data

dataset_dir = "split_directory"

# Define constants
num_classes = 27  # Number of mobile phone models
input_shape = (224, 224, 3)  # Input image dimensions
NUM_EPOCH = 10 
BATCH_SIZE = 32

# Load pre-trained MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x) #relu is activation function
predictions = Dense(num_classes, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions) #with the base MobileNet model's input and your custom classification layers as output

# Freeze pre-trained layers
for layer in base_model.layers:    # freeze the weights of the pre-trained MobileNet layers to prevent them from being updated during training.
    layer.trainable = False

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # optimizer is a algorithm,adam is commonly used optimiser.matrices evaluate the performance of model during traing an d testing

# Data augmentation and loading
train_datagen = ImageDataGenerator(   #image data generator for data augmentation during training,transformation like rascale,rotation etc
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(  #create a generator for training data,
    f'{dataset_dir}/train',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Validation data generator
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    f'./{dataset_dir}/val',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Train the model,
model.fit(   # train the model using the fit method, providing the training and validation generators, 
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=NUM_EPOCH,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

# Evaluate the model
test_generator = validation_datagen.flow_from_directory(   # evaluate the trained model on the test dataset using the evaluate method and print out the test accuracy.
    f'./{dataset_dir}/test',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Save the model
model.save('mobilenet_model.h5')
