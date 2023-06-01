import numpy as np
import csv
import matplotlib.pyplot as plt
import statistics
import os
import cv2

#pred spustenim kodu je potrebne naimportovat vsetky potrebne kniznice a OpenCV nainstalovat v terminali pomocou pip install opencv-python
def Analyzer(minor_min, minor_max):
    counter = 0
    counter_image = 0                           #inicializácia premenných
    major_axes = []
    y_append = []
    x_append = []
    counter_p = []
    counter_name = ["pocet najdenych = "]
    minor_append = []
    major_append = []
    angle_append = []
    grey_append = []
    ellipse_coordinates_area = []
    header = ['XC', 'YC', 'MINOR', 'MAJOR', "ANGLE", "AREA", "GREYLEVEL"]
    mean_name = ["Stredna hodnota = "]
    mean_n = []
    minor_axes = []
    file_list = []

    dir_path = "Img0002535"             #cesta k súboru z ktoreho sa obrazky nacitaju
    images = os.listdir(dir_path)  # najdenie vsetkych obrazkov v danom subore
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = [file for file in images if os.path.splitext(file)[-1].lower() in image_extensions]

    for file in image_files:                                #for cyklus na postupne nacitanie podla datumu
        filepath = os.path.join(dir_path, file)
        mod_time = os.path.getmtime(filepath)
        file_list.append((mod_time, file))
    sorted_files = sorted(file_list, key=lambda x: x[0])    #ulozenie obrazkov do sorted files

    for mod_time, file_name in sorted_files:
        if file_name.endswith('.bmp') or file_name.endswith('.png'):  # overenie ci obrazok je nacitany vo formáte bmp alebo png
            file_path = os.path.join(dir_path, file_name)  # vytvorenie celej cesty k obrázku
            img = cv2.imread(file_path)  # nacitanie obrazku pomocou opencv
            img_flip = cv2.flip(img, 0)
            img_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_grey = cv2.flip(img_grey, 0)
            gray = cv2.cvtColor(img_flip, cv2.COLOR_BGR2GRAY)
            blur = cv2.bilateralFilter(gray, 8, 80, 80)                 #vytvotrenie rozmazanej fotky
            ret = cv2.threshold(blur, 124, 255, cv2.THRESH_BINARY_INV)[1]   #vytvotrenie masky
            counter_image += 1
            contour, _ = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #nájdenie vsetkých kontúr na obrázku
            _, mask = cv2.threshold(img_grey, 124, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contour:                                                                   #for cyklus ktorý prechádza kazdú kontúru
                if len(c) >= 5:
                    ellipse = cv2.fitEllipse(c)                                                   #prisposobenie kazdeho tvaru na elipsu
                    (x, y), (major, minor), (angle) = ellipse                                   #parametre elipsy
                    major_axes.append(major)
                    minor, major = ellipse[1]

                    if minor <= (minor_max*2) and minor >=(minor_min*2):                        #podmienka, kde sa vykreslia všetky korektne stopy zelenou farbou
                        cv2.ellipse(img_flip, ellipse, (0, 255, 0), 2)                          #vyznačenie elipsy na obrázku
                        file_output = f"{dir_path}_output"
                        if not os.path.exists(file_output):
                            os.makedirs(file_output)

                        output_filename = f"{file_output}/processed_image_{file_name}.jpg"     #ulozenie obrázka s vyznacenými elipsami

                        cv2.imwrite(output_filename, img_flip)

                        mask = np.zeros_like(img_grey)
                        cv2.ellipse(mask, ellipse, 255, -1)                                             #vypocet odtiena sedej
                        min_val, max_val, _, _ = cv2.minMaxLoc(img_grey, mask=mask)
                        grey_append.append(max_val)
                        area = np.pi / 4 * major / 2 * minor / 2                        #vypocet plochy daneho tvaru
                        ellipse_coordinates_area.append(area)
                        #print("Vedlajsia os: " , minor / 2)
                        #print("Plocha: " , area)
                        #print("Odtieň sivej: " , max_val)

                        if major < minor:
                            minor_axes.append(major / 2)
                        else:
                            minor_axes.append(minor / 2)
                        counter += 1
                                                                                #podmienky posunu x a y súradnice podľa toho ako sa načítajú obrázky
                        if counter_image == 1:
                            y = y / 2
                            x = x / 2
                        elif counter_image == 2:
                            y = y / 2 + 1024
                            x = x / 2
                        elif counter_image == 3:
                            y = y / 2 + 1024 * 2
                            x = x / 2
                        elif counter_image == 4:
                            y = y / 2 + 1024 * 3
                            x = x / 2
                        elif counter_image == 5:
                            y = y / 2 + 1024 * 4
                            x = x / 2
                        elif counter_image == 6:
                            y = y / 2 + 1024 * 5
                            x = x / 2
                        elif counter_image == 7:
                            y = y / 2 + 1024 * 6
                            x = x / 2
                        elif counter_image == 8:
                            y = y / 2 + 1024 * 7
                            x = x / 2
                        elif counter_image == 9:
                            y = y / 2 + 1024 * 8
                            x = x / 2
                        elif counter_image == 10:
                            y = y / 2 + 1024 * 9
                            x = x / 2
                        elif counter_image == 11:
                            y = y / 2 + 1024 * 9
                            x = x / 2 + 1280
                        elif counter_image == 12:
                            y = y / 2 + 1024 * 8
                            x = x / 2 + 1280
                        elif counter_image == 13:
                            y = y / 2 + 1024 * 7
                            x = x / 2 + 1280
                        elif counter_image == 14:
                            y = y / 2 + 1024 * 6
                            x = x / 2 + 1280
                        elif counter_image == 15:
                            y = y / 2 + 1024 * 5
                            x = x / 2 + 1280
                        elif counter_image == 16:
                            y = y / 2 + 1024 * 4
                            x = x / 2 + 1280
                        elif counter_image == 17:
                            y = y / 2 + 1024 * 3
                            x = x / 2 + 1280
                        elif counter_image == 18:
                            y = y / 2 + 1024 * 2
                            x = x / 2 + 1280
                        elif counter_image == 19:
                            y = y / 2 + 1024 * 1
                            x = x / 2 + 1280
                        elif counter_image == 20:
                            y = y / 2
                            x = x / 2 + 1280
                        elif counter_image == 21:
                            y = y / 2
                            x = x / 2 + 1280 * 2
                        elif counter_image == 22:
                            y = y / 2 + 1024
                            x = x / 2 + 1280 * 2
                        elif counter_image == 23:
                            y = y / 2 + 1024 * 2
                            x = x / 2 + 1280 * 2
                        elif counter_image == 24:
                            y = y / 2 + 1024 * 3
                            x = x / 2 + 1280 * 2
                        elif counter_image == 25:
                            y = y / 2 + 1024 * 4
                            x = x / 2 + 1280 * 2
                        elif counter_image == 26:
                            y = y / 2 + 1024 * 5
                            x = x / 2 + 1280 * 2
                        elif counter_image == 27:
                            y = y / 2 + 1024 * 6
                            x = x / 2 + 1280 * 2
                        elif counter_image == 28:
                            y = y / 2 + 1024 * 7
                            x = x / 2 + 1280 * 2
                        elif counter_image == 29:
                            y = y / 2 + 1024 * 8
                            x = x / 2 + 1280 * 2
                        elif counter_image == 30:
                            y = y / 2 + 1024 * 9
                            x = x / 2 + 1280 * 2
                        elif counter_image == 31:
                            y = y / 2 + 1024 * 9
                            x = x / 2 + 1280 * 3
                        elif counter_image == 32:
                            y = y / 2 + 1024 * 8
                            x = x / 2 + 1280 * 3
                        elif counter_image == 33:
                            y = y / 2 + 1024 * 7
                            x = x / 2 + 1280 * 3
                        elif counter_image == 34:
                            y = y / 2 + 1024 * 6
                            x = x / 2 + 1280 * 3
                        elif counter_image == 35:
                            y = y / 2 + 1024 * 5
                            x = x / 2 + 1280 * 3
                        elif counter_image == 36:
                            y = y / 2 + 1024 * 4
                            x = x / 2 + 1280 * 3
                        elif counter_image == 37:
                            y = y / 2 + 1024 * 3
                            x = x / 2 + 1280 * 3
                        elif counter_image == 38:
                            y = y / 2 + 1024 * 2
                            x = x / 2 + 1280 * 3
                        elif counter_image == 39:
                            y = y / 2 + 1024 * 1
                            x = x / 2 + 1280 * 3
                        elif counter_image == 40:
                            y = y / 2
                            x = x / 2 + 1280 * 3
                        elif counter_image == 41:
                            y = y / 2
                            x = x / 2 + 1280 * 4
                        elif counter_image == 42:
                            y = y / 2 + 1024
                            x = x / 2 + 1280 * 4
                        elif counter_image == 43:
                            y = y / 2 + 1024 * 2
                            x = x / 2 + 1280 * 4
                        elif counter_image == 44:
                            y = y / 2 + 1024 * 3
                            x = x / 2 + 1280 * 4
                        elif counter_image == 45:
                            y = y / 2 + 1024 * 4
                            x = x / 2 + 1280 * 4
                        elif counter_image == 46:
                            y = y / 2 + 1024 * 5
                            x = x / 2 + 1280 * 4
                        elif counter_image == 47:
                            y = y / 2 + 1024 * 6
                            x = x / 2 + 1280 * 4
                        elif counter_image == 48:
                            y = y / 2 + 1024 * 7
                            x = x / 2 + 1280 * 4
                        elif counter_image == 49:
                            y = y / 2 + 1024 * 8
                            x = x / 2 + 1280 * 4
                        elif counter_image == 50:
                            y = y / 2 + 1024 * 9
                            x = x / 2 + 1280 * 4
                        elif counter_image == 51:
                            y = y / 2 + 1024 * 9
                            x = x / 2 + 1280 * 5
                        elif counter_image == 52:
                            y = y / 2 + 1024 * 8
                            x = x / 2 + 1280 * 5
                        elif counter_image == 53:
                            y = y / 2 + 1024 * 7
                            x = x / 2 + 1280 * 5
                        elif counter_image == 54:
                            y = y / 2 + 1024 * 6
                            x = x / 2 + 1280 * 5
                        elif counter_image == 55:
                            y = y / 2 + 1024 * 5
                            x = x / 2 + 1280 * 5
                        elif counter_image == 56:
                            y = y / 2 + 1024 * 4
                            x = x / 2 + 1280 * 5
                        elif counter_image == 57:
                            y = y / 2 + 1024 * 3
                            x = x / 2 + 1280 * 5
                        elif counter_image == 58:
                            y = y / 2 + 1024 * 2
                            x = x / 2 + 1280 * 5
                        elif counter_image == 59:
                            y = y / 2 + 1024 * 1
                            x = x / 2 + 1280 * 5
                        elif counter_image == 60:
                            y = y / 2
                            x = x / 2 + 1280 * 5
                        elif counter_image == 61:
                            y = y / 2
                            x = x / 2 + 1280 * 6
                        elif counter_image == 62:
                            y = y / 2 + 1024
                            x = x / 2 + 1280 * 6
                        elif counter_image == 63:
                            y = y / 2 + 1024 * 2
                            x = x / 2 + 1280 * 6
                        elif counter_image == 64:
                            y = y / 2 + 1024 * 3
                            x = x / 2 + 1280 * 6
                        elif counter_image == 65:
                            y = y / 2 + 1024 * 4
                            x = x / 2 + 1280 * 6
                        elif counter_image == 66:
                            y = y / 2 + 1024 * 5
                            x = x / 2 + 1280 * 6
                        elif counter_image == 67:
                            y = y / 2 + 1024 * 6
                            x = x / 2 + 1280 * 6
                        elif counter_image == 68:
                            y = y / 2 + 1024 * 7
                            x = x / 2 + 1280 * 6
                        elif counter_image == 69:
                            y = y / 2 + 1024 * 8
                            x = x / 2 + 1280 * 6
                        elif counter_image == 70:
                            y = y / 2 + 1024 * 9
                            x = x / 2 + 1280 * 6
                        elif counter_image == 71:
                            y = y / 2 + 1024 * 9
                            x = x / 2 + 1280 * 7
                        elif counter_image == 72:
                            y = y / 2 + 1024 * 8
                            x = x / 2 + 1280 * 7
                        elif counter_image == 73:
                            y = y / 2 + 1024 * 7
                            x = x / 2 + 1280 * 7
                        elif counter_image == 74:
                            y = y / 2 + 1024 * 6
                            x = x / 2 + 1280 * 7
                        elif counter_image == 75:
                            y = y / 2 + 1024 * 5
                            x = x / 2 + 1280 * 7
                        elif counter_image == 76:
                            y = y / 2 + 1024 * 4
                            x = x / 2 + 1280 * 7
                        elif counter_image == 77:
                            y = y / 2 + 1024 * 3
                            x = x / 2 + 1280 * 6
                        elif counter_image == 78:
                            y = y / 2 + 1024 * 2
                            x = x / 2 + 1280 * 6
                        elif counter_image == 79:
                            y = y / 2 + 1024 * 1
                            x = x / 2 + 1280 * 6
                        elif counter_image == 80:
                            y = y / 2
                            x = x / 2 + 1280 * 6
                        y_append.append(y)
                        x_append.append(x)
                        minor_append.append(minor/2)
                        major_append.append(major/2)
                        angle_append.append(angle)
                    elif minor > (minor_max * 2) and minor < (minor_min * 2) or major >= 0 and minor >= 0:              #else podmienka, kde sa vykreslia všetky zamietnuté cervenou farbou
                        cv2.ellipse(img_flip, ellipse, (0, 0, 255), 2)

    counter_p.append(counter)
    mean = statistics.mean(minor_append)
    mean_n.append(mean)
    plt.hist(minor_axes, bins=130, histtype='step', fill=True)                          #vykreslenie histogramu
    plt.xlim([3, 30])
    plt.xlabel('Velkost vedlajsej osi')
    plt.ylabel('Pocet objektov')
    plt.savefig(f'histogram_{dir_path}.png')
    plt.show()


    with open(f'ellipse_cordinates_{dir_path}', 'w',  newline='') as csvfile:                   #vytvorenie csv suboru zo vsetkych najdenych parametrov
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for value in range(len(x_append)):
            writer.writerow([x_append[value],y_append[value],minor_append[value],major_append[value],angle_append[value],ellipse_coordinates_area[value],grey_append[value]])
        writer.writerow(mean_name + mean_n)
        writer.writerow(counter_name + counter_p)


def main():                         #main kde volám funkciu Analyzer s parametrami minor_min a minor _max
    minor_min = 4
    minor_max = 40
    Analyzer(minor_min, minor_max)


if __name__ == '__main__':
    main()

"""
contours, _ = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                             #zakomentovaný záplavový filter

    idx = np.argmax([cv2.contourArea(cnt) for cnt in contours])
    cnt = contours[idx]

    thresh1 = cv2.fillPoly(ret, np.int32([cnt]),(255, 255, 255)) #Bug with fillPoly, needs explict cast to 32bit
    kernel = np.ones((0, 0), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_ELLIPSE, kernel, iterations=4)
    background = cv2.dilate(opening, kernel, iterations=5)

    dst = cv2.distanceTransform(opening, cv2.DIST_L2, 5, dstType=cv2.CV_32F)
    _, foreground = cv2.threshold(dst, 0.2 * dst.max(), 255, cv2.THRESH_BINARY)
    foreground = np.uint8(foreground)
    unknown = cv2.subtract(background, foreground)

    _, markers = cv2.connectedComponents(foreground)
    markers += 1
    markers[unknown == 255] = 0

    thresh = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(thresh, markers)

    markers = cv2.normalize(
        markers, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    _,thresh = cv2.threshold(markers, 9, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if len(c) >= 6:
            ellipse = cv2.fitEllipse(c)
            (xc, yc), (d1, d2), (angle) = ellipse
            if minor>60 or minor<70:
                cv2.ellipse(copy, ellipse, (0, 255, 0), 2)
            if minor>70:
                break

"""

