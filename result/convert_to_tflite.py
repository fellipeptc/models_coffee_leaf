# https://www.tensorflow.org/lite/convert/#convert_a_savedmodel_recommended_

import tensorflow as tf

# Load model
#load_model = tf.keras.models.load_model('D:/artigo_cafe/models/mobile_stop_model_v2.h5')
#load_model = tf.keras.models.load_model('D:/artigo_cafe/models/inceptionV3_stop_model_v2.h5')
#load_model = tf.keras.models.load_model('D:/artigo_cafe/models/resnet152_stop_model_v2.h5')
load_model = tf.keras.models.load_model('D:/artigo_cafe/models/vgg16_stop_model_v2.h5')

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(load_model)
modelo_tflite = converter.convert()

# Save the model
#arquivo_tflite = 'D:/artigo_cafe/result/tflite/mobile_stop_model_v2.tflite'
#arquivo_tflite = 'D:/artigo_cafe/result/tflite/inceptionV3_stop_model_v2.tflite'
#arquivo_tflite = 'D:/artigo_cafe/result/tflite/resnet152_stop_model_v2.tflite'
arquivo_tflite = 'D:/artigo_cafe/result/tflite/vgg16_stop_model_v2.tflite'

with open(arquivo_tflite, 'wb') as f:
    f.write(modelo_tflite)

print('Fim da conversão do modelo...')