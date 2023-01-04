import cv2
import numpy as np

scale = 1
delta = 0
ddepth = cv2.CV_16S

#Obtenemos la imagen de la carpeta 
src = cv2.imread('examen_b.tif');

#Filtro para el perfilado de la imagen 
img = cv2.detailEnhance(src)

#Recortamos la imagen del rostro
cropped_image = img[60:450, 100:550]

#Aplicamos las transformaciones necesarias para el filtro de sobel 
gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

#Obtenemos el centro de la imagen
image_center = tuple(np.array(grad.shape[1::-1]) / 2)

#La rotamos -20 grados con respecto al centro 
rot_mat = cv2.getRotationMatrix2D(image_center, -20, 1.0)

#Combinamos la matriz grad con la que hemos rotado con la funcion warpAffine
result = cv2.warpAffine(grad, rot_mat, grad.shape[1::-1], flags=cv2.INTER_LINEAR)

#Guardamos el resultado final 
cv2.imwrite('examen_b_resultado_final.tif',result) 
