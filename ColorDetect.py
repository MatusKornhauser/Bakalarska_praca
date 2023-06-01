import numpy as np
import cv2


def colorDetect():
    image = cv2.imread("Radon_/Image_1A.bmp")                       #hladanie pomocou farby, cesta na nacitanie jedneho obrázka
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_hue = np.array([0, 0, 0])
    upper_hue = np.array([0, 0, 65])

    mask = cv2.inRange(hsv, lower_hue, upper_hue)                   #maska daneho obrazka

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    #najdenie kontur
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x+10 + w, y+10 + h), (0, 255, 0), 2)

    cv2.imwrite("Colordetect.png", image)
    cv2.imwrite("maskcolor.png", mask)


def main():                     # main kde volám funkciu color detect
    colorDetect()


if __name__ == '__main__':
    main()