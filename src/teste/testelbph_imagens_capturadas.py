import cv2
import numpy as np
import os
from PIL import Image

detectorFacial = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt2.xml')
reconhecedorFacial = cv2.face.LBPHFaceRecognizer_create()
reconhecedorFacial.read('./classificador/LBPHYale.yml')
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFacial.detectMultiScale(imagemCinza,
                                                      scaleFactor=1.2,
                                                      minSize=(100, 100))
    for(x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedorFacial.predict(imagemFace)
        nome = ""
        if id == 1:
            nome = 'Cintia'
        if id == 2:
            nome = 'Gabrielle'
        elif id == 3:
            nome = "Anne"
        elif nome == 5:
            nome = "Debora"
        else:
            nome = "Undefined"
        cv2.putText(imagem, nome, (x, y +(a + 30)), font, 2, (0, 0, 255))
        cv2.putText(imagem, str(confianca), (x, y + (a+50)), font, 1, (0, 0, 255))

#para fechar a camera aperte a tecla q
    cv2.imshow('Face', imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
