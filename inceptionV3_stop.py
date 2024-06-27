import os
import tensorflow as tf

if tf.test.gpu_device_name():
    #List all available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Count Memory GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    except RuntimeError as e:
        print(e)
    print('Tensorflow irá usar a GPU, {}'.format(tf.test.gpu_device_name()))
else:
    # Number MAX threads CPU
    print('Tensorflow irá usar a CPU')
    os.environ['OMP_NUM_THREADS'] = '8' 

# Path dir images [train, teste, val]
train_data_dir = 'D:/artigo_cafe/images/data/training/'
test_data_dir = 'D:/artigo_cafe/images/data/test/'
val_data_dir = 'D:/artigo_cafe/images/data/validation/'

# Data
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
)

# Setup pre processing
image_size = (224, 224)
batch_size = 32
# Pre image processing
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
)
test_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
)
val_generator = datagen.flow_from_directory(
    val_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
)

# [sick, healthy]
train_classes = train_generator.num_classes
test_classes = test_generator.num_classes
val_classes = val_generator.num_classes
print("Qtd. class:", train_classes," - ", test_classes, " - ", val_classes)

# Load InceptionV3 pre trained (nothing dense layers)
from keras.applications import InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the Weights
base_model.trainable = False

# Create model using base the class Sequential()
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
num_classes = 2
model = Sequential() 
model.add(base_model) 

# Add layers
model.add(GlobalAveragePooling2D()) 
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))

# Exit layer
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

# Train Model
history = model.fit(    
    train_generator,
    epochs=200, 
    validation_data=val_generator,
    steps_per_epoch=128,  # Número de passos por época
    )

# Save file model .h5
model.save('D:/artigo_cafe/models_stop/inceptionV3_stop_model_v2.h5')
import pickle
with open('D:/artigo_cafe/models_stop/history/inceptionV3_stop_history_v2.pickle', 'wb') as file:
    pickle.dump(history.history, file)
    
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)