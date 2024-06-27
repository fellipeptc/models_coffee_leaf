import os
import cv2
import time
import numpy as np

# Diretório contendo as imagens
#dir_img = 'D:/artigo_cafe/images/data/bicho_mineiro/'
dir_img = 'D:/artigo_cafe/images/filters/original/saudavel'

# Diretório contendo as imagens com filtro salt
#dir_img_noise = 'D:/artigo_cafe/images/filters/noise_gau/bicho_mineiro'
dir_img_noise = 'D:/artigo_cafe/images/filters/noise_gau/saudavel'

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
        
        # Obter as dimensões da imagem
        altura, largura, canais = img.shape

        # Parâmetros para controlar o ruído
        intensidade_ruido = 70  # Intensidade do ruído 

        # Aplicando ruído gaussiano
        ruido = np.random.normal(0, intensidade_ruido, (altura, largura, canais))
        imagem_ruidosa = np.clip(img + ruido, 0, 255).astype(np.uint8)
   
        # Define o nome do novo arquivo
        #file_name = f"resize_{indice}.jpg"
        resize_name = f"noise_gau_s_{index}.jpg"
        new_path = os.path.join(dir_img_noise, resize_name)

        # Salva a imagem
        cv2.imwrite(new_path, imagem_ruidosa)
        # Incrementando valor na lista de imagens        
        index = index + 1
        time.sleep(0.1)
print('Fim do filtro gaussiano nas imagens...')