import time
import cv2
import os
import numpy as np

# Caminho para a pasta das imagens
#pasta_base = "D:/artigo_cafe/images/data/bicho_mineiro/"
#pasta_rotation = "D:/artigo_cafe/images/rotation/bicho_mineiro/"

pasta_base = "D:/artigo_cafe/images/data/saudavel/"
pasta_rotation = "D:/artigo_cafe/images/rotation/saudavel/"

# Ângulos para rotacionar as imagens
angulos = [45, 90, 135, 180, 225, 270, 315, 360]

# indice para o loop
indice = 0
# Percorre todos os arquivos na pasta
for arquivo in os.listdir(pasta_base):
    # Verifica se o arquivo é uma imagem (extensões suportadas)
    if arquivo.endswith(".jpg") or arquivo.endswith(".jpeg") or arquivo.endswith(".png"):
        # Abre a imagem
        caminho_imagem = os.path.join(pasta_base, arquivo)
        imagem = cv2.imread(caminho_imagem)
        print('imagem: ', caminho_imagem)

        # Rotaciona a imagem para cada ângulo especificado
        for angulo in angulos:
            # Obtém as dimensões da imagem
            altura, largura = imagem.shape[:2]

            # Calcula o centro da imagem
            centro = (largura // 2, altura // 2)

            # Realiza a rotação da imagem
            matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo, 1.0)
            imagem_rotacionada = cv2.warpAffine(imagem, matriz_rotacao, (largura, altura))

            # Define o nome do novo arquivo
            #nome_arquivo_rotacionado = f"fs_{angulo}_{indice}.jpg"
            nome_arquivo_rotacionado = f"bm_{angulo}_{indice}.jpg"
            caminho_arquivo_rotacionado = os.path.join(pasta_rotation, nome_arquivo_rotacionado)

            print('angulo: ', angulo, '\tindice: ', indice)

            # Salva a imagem rotacionada
            cv2.imwrite(caminho_arquivo_rotacionado, imagem_rotacionada)
            # Incrementando valor na lista de imagens        
            indice = indice + 1
            time.sleep(0.1)