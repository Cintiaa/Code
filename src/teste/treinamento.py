import cv2
import os
import numpy as np
from PIL import Image

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getIdComImagem():
    caminhos = [os.path.join('../databases/imagefaces', f) for f in os.listdir('../databases/imagefaces')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split(".")[1])
        print(id)
        ids.append(id)
        faces.append(imagemFace)
        cv2.imshow("face", imagemFace)
        cv2.waitKey(10)
    return np.array(ids), faces
ids, faces = getIdComImagem()


print('Realizando treinamento com as imagens!')

#criando classificador EigenFace
eigenface.train(faces, ids)
eigenface.write('../classificador/EigenFace.yml')

#criando classificador FisherFace
fisherface.train(faces, ids)
fisherface.write('../classificador/FisherFace.yml')


#criando classificador LBPH
lbph.train(faces, ids)
lbph.write('../classificador/LBPHFace.yml')


print('Treinamento realizado')


