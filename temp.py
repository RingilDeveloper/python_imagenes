import cv2
import numpy as np




fin=True

def menu():

	print ("""MENU
        1) girara la imagen 90 grados
        2) girar la imagen 180 grados
        3) dejar la imagen en el canal rojo
        4) dejar la imagen en el canal verde
        5) dejar la imagen en el canal azul
        6) detectar el color verde de la imagen
        7) suma de imagenes para superponerlar
        8) dejar la imagen en escala de grises
        9) pasar la imagen a blanco y negro
        10) negativo de la imagen en blanco y negro
        0) salir 
        """)
    
def imagen_90():
    imagen1 = cv2.imread('imagen.png')
    (a, l) = imagen1.shape[:2] # obtiene la anchura y altura
    centro = (a / 2, l / 2) # calcula el centro de la imagen
    M = cv2.getRotationMatrix2D(centro, 90, 1.0)
    rotar90 = cv2.warpAffine(imagen1, M, (a, l))#imagen rotada
    cv2.imwrite('rotated90.jpg',rotar90)##guardar imagen
    cv2.imshow('Image rotated by 90 degrees',rotar90)
    cv2.waitKey(0) # Espera hasta que el usuario pulse una tecla en la ventana
    cv2.destroyAllWindows()
def imagen_180():
    imagen2 = cv2.imread('imagen.png')
    (a, l) = imagen2.shape[:2] # obtiene la anchura y altura
    centro = (a / 2, l / 2) # calcula el centro de la imagen
    M = cv2.getRotationMatrix2D(centro, 180, 1.0)
    rotar180 = cv2.warpAffine(imagen2, M, (a, l))#imagen rotada
    cv2.imwrite('rotated180.jpg',rotar180)##guardar imagen
    cv2.imshow('Image rotated by 180 degrees',rotar180)
    cv2.waitKey(0) # Espera hasta que el usuario pulse una tecla en la ventana
    cv2.destroyAllWindows()
def imagen_rojo():
    image = cv2.imread("imagen.png")
    r = image.copy()
    r[:, :, 0] = 0#se pone el canal ver y el azul en 0
    r[:, :, 1] = 0
    cv2.imshow("R-RGB", r)## muetsra la imagen
    cv2.waitKey(0)
def imagen_azul():
    image = cv2.imread("imagen.png")
    b = image.copy()
    b[:, :, 1] = 0
    b[:, :, 2] = 0
    cv2.imshow("R-RGB", b)
    cv2.waitKey(0)
def imagen_verde():
    image = cv2.imread("imagen.png")
    g = image.copy()
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    cv2.imshow("R-RGB", g)
    cv2.waitKey(0)
def detectar():
    verde_bajos = np.array([49,50,50])
    verde_altos = np.array([80, 255, 255])
    imagen = cv2.imread("paisaje.jpg")
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, verde_bajos, verde_altos)
    cv2.imshow('mascara', mask)
    cv2.imshow('Camara', imagen)
    cv2.waitKey(0)
def overlay(image1, image2, x, y):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(image1_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image1_mask = np.zeros_like(image1)
    cv2.drawContours(image1_mask, contours, -1, (255,255,255), -1)
    idx = np.where(image1_mask == 255)
    image2[y+idx[0], x+idx[1], idx[2]] = image1[idx[0], idx[1], idx[2]]
    return image2
def grises():
    img = cv2.imread('imagen.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
def negro ():
    img = cv2.imread('imagen.png')
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Black white image', blackAndWhiteImage)
    cv2.waitKey(0)
def negative ():
    img = cv2.imread('imagen.png')
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    img_not = cv2.bitwise_not(blackAndWhiteImage)
    cv2.imshow("Invert1",img_not)
    cv2.waitKey(0)

while fin == True:

	menu()

	smenu = int(input("inserta un opcion: "))  # solicituamos una opcion
    
        if smenu == 1:
            imagen_90()
        elif smenu == 2:
            imagen_180()
        elif smenu == 3:
            imagen_rojo()
        elif smenu == 4:
            imagen_verde()
        elif smenu == 5:
            imagen_azul()
        elif smenu == 6:
            detectar()
        elif smenu == 7:
            paisaje = cv2.imread("paisaje.jpg")
            st = cv2.imread("imagen.png")
            overlayed = overlay(st, paisaje, 300, 300)
            cv2.imwrite("overlayed.png", overlayed)
            cv2.imshow('Camara', overlayed)
            cv2.waitKey(0)
        elif smenu == 8:
            grises()
        elif smenu == 9:
            negro()
        elif smenu == 10:
            negative()
        elif smenu == 0:
            fin = False
    	else:

            ("opcion incorrecta ")

		
    

     