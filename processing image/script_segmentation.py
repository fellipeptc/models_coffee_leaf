import os
import cv2
import time
import numpy as np

# Diretório contendo as imagens
#dir_img = 'D:/artigo_cafe/images/data/bicho_mineiro/'
dir_img = 'D:/artigo_cafe/images/data/saudavel/'

# Diretório contendo as imagens
#dir_img_segmentation = 'D:/artigo_cafe/images/filters/segmentation/bicho_mineiro'
dir_img_segmentation = 'D:/artigo_cafe/images/filters/segmentation/saudavel'

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
    
        # Converter para espaço de cor HSV
        imagem_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Definir limites para tons de verde
        limite_inferior = np.array([30, 40, 40])
        limite_superior = np.array([93, 255, 255])
        # Criar uma máscara para a faixa de verde
        mascara = cv2.inRange(imagem_hsv, limite_inferior, limite_superior)

        # Aplicar a máscara à imagem original
        imagem_segmentada = cv2.bitwise_and(img, img, mask=mascara)
   
        # Define o nome do novo arquivo
        #file_name = f"resize_{indice}.jpg"
        resize_name = f"segmentation_s_{index}.jpg"
        new_path = os.path.join(dir_img_segmentation, resize_name)

        # Salva a imagem
        cv2.imwrite(new_path, imagem_segmentada)
        # Incrementando valor na lista de imagens        
        index = index + 1
        time.sleep(0.1)
print('Fim do filtro de segmentação nas imagens...')