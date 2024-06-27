import os
import tensorflow as tf

if tf.test.gpu_device_name():
    #Listando todas as GPUs disponíveis
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Limitar a quantidade de memória da GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    except RuntimeError as e:
        print(e)
    print('Tensorflow irá usar a GPU, {}'.format(tf.test.gpu_device_name()))
else:
    # Define o número máximo de threads da CPU
    print('Tensorflow irá usar a CPU')
    os.environ['OMP_NUM_THREADS'] = '8' 

# Caminho para o diretório das imagens [train, teste, val]
train_data_dir = 'D:/artigo_cafe/images/data/training/'
test_data_dir = 'D:/artigo_cafe/images/data/test/'
val_data_dir = 'D:/artigo_cafe/images/data/validation/'

# Parâmetros para pré-processamento e aumento de dados
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
)

# Configurações do pré-processamento
image_size = (224, 224)
batch_size = 32
# Carregar e pré-processar as imagens
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

# [bicho_mineiro, saudavel]
train_classes = train_generator.num_classes
test_classes = test_generator.num_classes
val_classes = val_generator.num_classes
print("Qtd. class:", train_classes," - ", test_classes, " - ", val_classes)

from keras.applications import MobileNet
# Carregar o modelo MobileNet pré-treinado (sem a camada de classificação)
mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar os pesos do MobileNet para evitar o treinamento
mobilenet_model.trainable = False


# Criar um novo modelo com base na VGG16 pré-treinada e usando Sequential()
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
num_classes = 2
model = Sequential() 
model.add(mobilenet_model) 

# Adicionar camadas convolutivas adicionais
model.add(MaxPooling2D((2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))

# Adicionar uma camada de saída densa de classificação
# 2 classes: "saudável" e "bicho mineiro"
model.add(Dense(2, activation='softmax')) 

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Definir o callback EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

# Treinando o modelo
history = model.fit(
    train_generator,
    epochs=200,
    validation_data=val_generator,
    steps_per_epoch=128,  # Número de passos por época
    )
# Salvando o modelo
model.save('D:/artigo_cafe/models_stop/mobile_stop_model_v2.h5')
# Salvar o histórico em um arquivo
import pickle
with open('D:/artigo_cafe/models_stop/history/mobile_stop_history_v2.pickle', 'wb') as file:
    pickle.dump(history.history, file)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

