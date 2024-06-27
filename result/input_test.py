import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if tf.test.gpu_device_name():
    #Listando todas as GPUs disponíveis
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Limitar a quantidade de memória da GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3069)])
    except RuntimeError as e:
        print(e)
    print('Tensorflow irá usar a GPU, {}'.format(tf.test.gpu_device_name()))
else:
    # Define o número máximo de threads da CPU
    print('Tensorflow irá usar a CPU')
    os.environ['OMP_NUM_THREADS'] = '8' 

#TODO - Todos os modelos gerados
# Carregar o modelos salvos
#vgg16_model = tf.keras.models.load_model('D:/artigo_cafe/vgg16_model.h5')
#inceptionV3_model = tf.keras.models.load_model('D:/artigo_cafe/models/inceptionV3_model.h5')
#resnet152_model = tf.keras.models.load_model('D:/artigo_cafe/models/resnet152_model.h5')

mobilenet_model = tf.keras.models.load_model('D:/artigo_cafe/models/mobile_stop_model_v2.h5')

# Caminho das imagens de teste
test_path = 'D:/artigo_cafe/images/data/test/'

# Carregando as imagens para testar o modelo
from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

#TODO - Alterar o modelo para exportar os dados
# Realizando as previsões com as imagens
predictions = mobilenet_model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
certainty = np.max(predictions, axis=1)
class_labels = test_data.class_indices

# Utilizado para extrair informaçoes e salvar em .csv
file_names_list = []
probability_list = []
predicted_class_list = []
real_class_list = []

for i in range(len(test_data.filenames)):
    image_path = test_data.filepaths[i]
    image_label = test_data.labels[i]
    predicted_label = predicted_classes[i]
    predicted_class = list(class_labels.keys())[list(class_labels.values()).index(predicted_label)]
    probability = certainty[i]

    # Gerando a lista das imagens com a probabilidade de cada
    file_names_list.append(test_data.filenames[i])
    probability_list.append(probability)
    predicted_class_list.append(predicted_class)
    real_class_list.append(list(class_labels.keys())[list(class_labels.values()).index(test_data.classes[i])])

    print("Imagem:", image_path)
    print("Classe real:", test_data.classes[i])
    print("Classe prevista:", predicted_class)
    print("Certeza:", probability)
    print()

#TODO - Alterar o nome do arquivo .csv
# Caminho do arquivo CSV de saída
import csv
caminho_arquivo_csv = 'D:/artigo_cafe/result/data/mobilenet_v2.csv'

# Criar o arquivo CSV
with open(caminho_arquivo_csv, 'w', newline='') as arquivo_csv:
    writer = csv.writer(arquivo_csv, delimiter=';')
    writer.writerow(['Nome da Imagem', 'Probabilidade', 'Classe', 'Classe real'])  # Escrever o cabeçalho
    
    # Escrever os dados de cada imagem
    for nome, probabilidade, classe, classe_real in zip(file_names_list, probability_list, predicted_class_list, real_class_list):
        writer.writerow([nome, probabilidade, classe, classe_real])
        
print('Dados salvos em CSV com sucesso.')