import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image

img = cv2.imread('3.jpg',cv2.IMREAD_COLOR) # <----
cv2.imshow('Original',img)

# Redimensiona a imagem
img = cv2.resize(img, (620,480) )
cv2.imshow('Redimensionada',img)

# Cinza
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('Cinza', cinza)

# Borra para reduzir o ruido
cinza = cv2.bilateralFilter(cinza, 11, 17, 17) 
cv2.imshow('Borrada', cinza)

# Bordas
bordas = cv2.Canny(cinza, 30, 200) 
cv2.imshow('Bordas', bordas)

cnts = cv2.findContours(bordas.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
t_contornos = None

# Laco nos contornos
for c in cnts:
    # Aproxima o contorno
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
    # Se o contorno tem 4 pontos pressupoe seque localizou a placa
    if len(approx) == 4:
        t_contornos = approx
        break

if t_contornos is None:
    detected = 0
    print("Sem contorno detectado")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(img, [t_contornos], -1, (0, 255, 0), 3)
cv2.imshow('Contornos',img)

# Mascara
mascara = np.zeros(cinza.shape, np.uint8)
nova_imagem = cv2.drawContours(mascara, [t_contornos], 0, 255, -1,)
nova_imagem = cv2.bitwise_and(img, img, mask=mascara)
cv2.imshow('Mascara', nova_imagem)

# Corte
(x, y) = np.where(mascara == 255)
(topox, topoy) = (np.min(x), np.min(y))
(infx, infy) = (np.max(x), np.max(y))
corte = cinza[topox:infx+1, topoy:infy+1]
cv2.imshow('Corte', corte)

# Le a placa
placa = pytesseract.image_to_string(corte, config='--psm 11')
print("A placa é:", placa)

cv2.waitKey(0)
cv2.destroyAllWindows()
