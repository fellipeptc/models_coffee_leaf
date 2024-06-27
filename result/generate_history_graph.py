import pickle
import matplotlib.pyplot as plt

# TODO
# Carregar o histórico de um arquivo com caminho personalizado
#historico_arquivo = 'D:/artigo_cafe/models/history/vgg16_stop_history_v2.pickle'
#historico_arquivo = 'D:/artigo_cafe/models/history/resnet152_stop_history_v2.pickle'
historico_arquivo = 'D:/artigo_cafe/models/history/mobile_stop_history_v2.pickle'
#historico_arquivo = 'D:/artigo_cafe/models/history/inceptionV3_stop_history_v2.pickle'

with open(historico_arquivo, 'rb') as file:
    loaded_history = pickle.load(file)

print('Plotando gráficos para analisar...')

# Plotar a perda durante o treinamento
# TODO Mudar nome do título
title = 'MobileNet'
plt.plot(loaded_history['loss'], label='Training Loss')
plt.plot(loaded_history['val_loss'], label='Validation Loss')
plt.plot(loaded_history['val_loss'][-1], label='Test Loss')
plt.title(f'Loss value using {title} model as base')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotar a acurácia durante o treinamento
import matplotlib.pyplot as plt
plt.plot(loaded_history['accuracy'])
plt.plot(loaded_history['val_accuracy'])
plt.plot(loaded_history['val_accuracy'][-1])
plt.title(f'Accuracy value using {title} model as base')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
plt.show()

print('Fim do script...')