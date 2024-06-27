import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

#TODO - Todos os modelos gerados
# Carregar o modelos salvos com parada
mobilenet_model = tf.keras.models.load_model('D:/artigo_cafe/models/mobile_stop_model_v2.h5')
#inceptionV3_model = tf.keras.models.load_model('D:/artigo_cafe/models/inceptionV3_stop_model_v2.h5')
#resnet152_stop_model_v2 = tf.keras.models.load_model('D:/artigo_cafe/models/resnet152_stop_model_v2.h5')
#vgg16_stop_model_v2 = tf.keras.models.load_model('D:/artigo_cafe/models/vgg16_stop_model_v2.h5')



# Caminho das imagens de teste
test_path = 'D:/artigo_cafe/images/data/test/'

# Definir as classes
classes = ['bicho_mineiro', 'saudavel']

# Inicializar os contadores
truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0

# Percorrer todas as imagens de teste
for class_name in classes:
    class_path = os.path.join(test_path, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)

        # Carregar a imagem e redimensionar para o tamanho esperado pelo modelo
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Normalizar a imagem
        image = image / 255.0

        #TODO - Alterar variavel para usar outros modelos
        # Fazer a previsão usando o modelo
        predictions = mobilenet_model.predict(image)
        predicted_class = np.argmax(predictions)

        # Comparar a previsão com o rótulo verdadeiro
        true_class = classes.index(class_name)
        if predicted_class == true_class and true_class == 0: #0 para bicho_meiro e 1 para saudavel
            truePositive += 1
        if predicted_class != true_class and true_class == 0:
            falseNegative += 1
        if predicted_class != true_class and true_class == 1:
            falsePositive += 1
        if predicted_class == true_class and true_class == 1:
            trueNegative += 1

# Imprimir os valores de TP, FP e FN
print('True Positives:', truePositive)
print('False Positives:', falsePositive)
print('True Negative:', trueNegative)
print('False Negatives:', falseNegative)

# Calcular a precisão
precision = truePositive / (truePositive + falsePositive)
# Calcular o recall
recall = truePositive / (truePositive + falseNegative)
# Calcular a medida F1
f1 = 2 * (precision * recall) / (precision + recall)
print("Medida F1:", f1)
print()

