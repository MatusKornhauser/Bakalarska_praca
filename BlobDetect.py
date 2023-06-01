import numpy as np
import cv2


def blobDetection():                                    #metoda hladania podla kruhov
    image = cv2.imread("Radon_/Image_1A.bmp")                       #cesta na nacitanie jedneho obrazka

    params = cv2.SimpleBlobDetector_Params()                #parametre daneho kruhu
                                                            #maska
    params.minThreshold = 1
    params.maxThreshold = 400
                                                                #plocha
    params.filterByArea = True
    params.minArea = 60

                                                                    # kruhovitost
    params.filterByCircularity = True
    params.minCircularity = 0.1

    params.filterByConvexity = True
    params.minConvexity = 0.1

    params.filterByInertia = True
    params.minInertiaRatio = 0.001
    params.maxInertiaRatio = 1
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)

    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 250), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite("Blobdetect.png", blobs)


def main():  # main kde volám funkciu blobDetect
    blobDetection()


if __name__ == '__main__':
    main()

