import os
import cv2
import time
import numpy as np

# Diretório contendo as imagens
#dir_img = 'D:/artigo_cafe/images/data/bicho_mineiro/'
dir_img = 'D:/artigo_cafe/images/data/saudavel/'

# Diretório contendo as imagens com filtro salt
#dir_img_solarize = 'D:/artigo_cafe/images/filters/solarize/bicho_mineiro'
dir_img_solarize = 'D:/artigo_cafe/images/filters/solarize/saudavel'

# Função para aplicar o efeito de solarização
def solarize(imagem, threshold=150):
    # Inverte os pixels que têm intensidade maior que o limiar (threshold)
    return np.where(imagem < threshold, imagem, 255 - imagem)

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
        
        # Defina o limiar de solarização
        limiar_solarizacao = 196

        # Aplicar a função de solarização à imagem
        imagem_solarizada = solarize(img, limiar_solarizacao)
   
        # Define o nome do novo arquivo
        #file_name = f"resize_{indice}.jpg"
        resize_name = f"solarize_s{index}.jpg"
        new_path = os.path.join(dir_img_solarize, resize_name)

        # Salva a imagem
        cv2.imwrite(new_path, imagem_solarizada)
        # Incrementando valor na lista de imagens        
        index = index + 1
        time.sleep(0.1)
print('Fim do filtro de solarização nas imagens...')