import os
import cv2
import time
import numpy as np

# Diretório contendo as imagens
#dir_img = 'D:/artigo_cafe/images/data/bicho_mineiro/'
dir_img = 'D:/artigo_cafe/images/data/saudavel/'

# Diretório contendo as imagens
#dir_img_linear = 'D:/artigo_cafe/images/filters/linear_contrast/bicho_mineiro'
dir_img_linear = 'D:/artigo_cafe/images/filters/linear_contrast/saudavel'

# Função para aplicar o filtro de contraste linear
def linear_contrast(image, alpha, beta):
    # Aplicar a transformação linear para ajustar o contraste
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

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
    
        # Parâmetros do filtro de contraste linear (ajuste conforme necessário)
        alpha = 1.8  # Fator de contraste
        beta = 0.5     # Deslocamento (0 mantém o brilho original)

        # Aplicar o filtro de contraste linear à imagem
        linear_contrast_image = linear_contrast(img, alpha, beta)
   
        # Define o nome do novo arquivo
        #file_name = f"resize_{indice}.jpg"
        resize_name = f"linear_contrast_s_{index}.jpg"
        new_path = os.path.join(dir_img_linear, resize_name)

        # Salva a imagem
        cv2.imwrite(new_path, linear_contrast_image)
        # Incrementando valor na lista de imagens        
        index = index + 1
        time.sleep(0.1)
print('Fim do filtro constraste linear nas imagens...')