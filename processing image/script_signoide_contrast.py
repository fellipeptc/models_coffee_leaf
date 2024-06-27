import os
import cv2
import time
import numpy as np

# Diretório contendo as imagens
#dir_img = 'D:/artigo_cafe/images/data/bicho_mineiro/'
dir_img = 'D:/artigo_cafe/images/data/saudavel/'

# Diretório contendo as imagens com contraste sigmoidal
#dir_img_linear = 'D:/artigo_cafe/images/filters/signoid_contrast/bicho_mineiro'
dir_img_linear = 'D:/artigo_cafe/images/filters/signoid_contrast/saudavel'

# Função para aplicar o filtro de contraste sigmoidal
def sigmoid_contrast(imagem, k, x0):
    # Aplicar a função sigmoidal aos valores dos pixels
    return 255 / (1 + np.exp(-k * (imagem - x0)))

# indice para o loop
index = 0

# Percorre todos os arquivos na pasta
for file in os.listdir(dir_img):
    # Verifica se o arquivo é uma imagem (extensões suportadas)
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        # Abre a imagem
        img_path = os.path.join(dir_img, file)
        img = cv2.imread(img_path)
        print('Image: ', img_path)
    
        # Parâmetros do filtro sigmoidal (ajuste conforme necessário)
        k = 0.25 # Controle da inclinação da curva (maior valor aumenta o contraste)
        x0 = 130  # Ponto de inflexão (valor que não será afetado)

        # Aplicar o filtro de contraste sigmoidal à imagem
        sigmoid_contrast_image = sigmoid_contrast(img, k, x0).astype(np.uint8)
   
        # Define o nome do novo arquivo
        #file_name = f"resize_{indice}.jpg"
        resize_name = f"signoid_contrast_s_{index}.jpg"
        new_path = os.path.join(dir_img_linear, resize_name)

        # Salva a imagem
        cv2.imwrite(new_path, sigmoid_contrast_image)
        # Incrementando valor na lista de imagens        
        index = index + 1
        time.sleep(0.1)
print('Fim do filtro constraste sigmoidal nas imagens...')