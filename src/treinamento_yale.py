import cv2
import os
import numpy as np
from PIL import Image

eigenface = cv2.face.EigenFaceRecognizer_create(30, 9000)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImagemComId():
    caminhos = [os.path.join('databases/yalefaces/treinamento', f) for f in os.listdir('databases/yalefaces/treinamento')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = Image.open(caminhoImagem).convert('L') #converte para scala de cinza
        imagemNP = np.array(imagemFace, 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
        ids.append(id)
        faces.append(imagemNP)

    return np.array(ids), faces

ids, faces = getImagemComId()

print('Realizando treinamento com as imagens!')

#criando classificador EigenFace
eigenface.train(faces, ids)
eigenface.write('classificador/EigenYale.yml')

#criando classificador FisherFace
fisherface.train(faces, ids)
fisherface.write('classificador/FisherYale.yml')


#criando classificador LBPH
lbph.train(faces, ids)
lbph.write('classificador/LBPHYale.yml')


print('Treinamento realizado')


