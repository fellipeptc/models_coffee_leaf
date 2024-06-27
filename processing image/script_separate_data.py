import os
import random
import shutil

# Diretório contendo as imagens
image_dir = 'D:/artigo_cafe/images/rotation/'

# Diretório para salvar as imagens divididas
train_dir = 'D:/artigo_cafe/images/data/training'
test_dir = 'D:/artigo_cafe/images/data/test'
val_dir = 'D:/artigo_cafe/images/data/validation'

# Porcentagem para treinamento, teste e validação
train_percent = 0.7
test_percent = 0.15
val_percent = 0.15

# Criar os diretórios de treinamento, teste e validação
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Lista de classes
classes = ['bicho_mineiro', 'saudavel']

print('Divisao das imagens em (70% train) - (15% test) - (15% - val)')
# Iterar sobre cada classe
for class_name in classes:
    class_dir = os.path.join(image_dir, class_name)
    images = os.listdir(class_dir)
    num_images = len(images)
    num_train = int(train_percent * num_images)
    num_test = int(test_percent * num_images)
    num_val = int(val_percent * num_images)

    # Embaralhar a lista de imagens
    print('Fazendo o embaralhamento das imagens ', class_name)
    random.shuffle(images)

    # Dividir as imagens em treinamento, teste e validação
    train_images = images[:num_train]
    test_images = images[num_train:num_train + num_test]
    val_images = images[num_train + num_test:]

    # Copiar as imagens para os diretórios correspondentes
    for image in train_images:
        src = os.path.join(class_dir, image)
        dst = os.path.join(train_dir, class_name, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for image in test_images:
        src = os.path.join(class_dir, image)
        dst = os.path.join(test_dir, class_name, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for image in val_images:
        src = os.path.join(class_dir, image)
        dst = os.path.join(val_dir, class_name, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

print('Fim da divisao de imagens...')