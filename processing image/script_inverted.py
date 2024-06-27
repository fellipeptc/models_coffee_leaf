import os
import cv2
import time
import numpy as np

# Diretório contendo as imagens
dir_img = 'D:/artigo_cafe/images/data/bicho_mineiro/'
#dir_img = 'D:/artigo_cafe/images/data/saudavel/'

# Diretório contendo as imagens
dir_img_inverted = 'D:/artigo_cafe/images/filters/inverted/bicho_mineiro'
#dir_img_inverted = 'D:/artigo_cafe/images/filters/inverted/saudavel'

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

        # Inverter a imagem
        imagem_invertida = 255 - img
   
        # Define o nome do novo arquivo
        #file_name = f"resize_{indice}.jpg"
        resize_name = f"inverted_bm_{index}.jpg"
        new_path = os.path.join(dir_img_inverted, resize_name)

        # Salva a imagem
        cv2.imwrite(new_path, imagem_invertida)
        # Incrementando valor na lista de imagens        
        index = index + 1
        time.sleep(0.1)
print('Fim do filtro invertido nas imagens...')