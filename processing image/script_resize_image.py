import os
import cv2
import time

# Diretório contendo as imagens (TROQUE AS PASTAS)
# dir_img = 'D:/artigo_cafe/images/samsung/bicho_mineiro/'
dir_img = 'D:/artigo_cafe/images/samsung/saudavel/'

# Diretório contendo as imagens redimensionadas
# dir_img_resize = 'D:/artigo_cafe/images/resize/bicho_mineiro/'
dir_img_resize = 'D:/artigo_cafe/images/resize/saudavel/'


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
        print('Original Dimensions : ',img.shape)

        scale_percent = 20 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        # Define o nome do novo arquivo
        #file_name = f"resize_{indice}.jpg"
        resize_name = f"resize_s_{index}.jpg"
        new_path = os.path.join(dir_img_resize, resize_name)

        # Salva a imagem rotacionada
        cv2.imwrite(new_path, resized)
        # Incrementando valor na lista de imagens        
        index = index + 1
        time.sleep(0.1)
print('Fim do redimensionamento de imagens...')