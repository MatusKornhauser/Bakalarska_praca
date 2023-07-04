import numpy as np
import csv
import matplotlib.pyplot as plt
import statistics
import os
import cv2

#pred spustenim kodu je potrebne naimportovat vsetky potrebne kniznice a OpenCV nainstalovat v terminali pomocou pip install opencv-python

class ImageAnalyzer:
    def __init__(self, minor_min, minor_max):
        self.minor_min = minor_min
        self.minor_max = minor_max
        self.counter = 0
        self.counter_image = 0
        self.major_axes = []
        self.y_append = []
        self.x_append = []
        self.counter_p = []
        self.counter_name = ["pocet najdenych = "]
        self.minor_append = []
        self.major_append = []
        self.angle_append = []
        self.grey_append = []
        self.ellipse_coordinates_area = []
        self.header = ['XC', 'YC', 'MINOR', 'MAJOR', "ANGLE", "AREA", "GREYLEVEL"]
        self.mean_name = ["Stredna hodnota = "]
        self.mean_n = []
        self.minor_axes = []
        self.file_list = []

    def analyze_images(self, dir_path):
        images = os.listdir(dir_path)                                                                                 #najdenie vsetkych obrazkov v danom subore
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        image_files = [file for file in images if os.path.splitext(file)[-1].lower() in image_extensions]

        for file in image_files:                                                                                    #for cyklus na postupne nacitanie podla datumu
            filepath = os.path.join(dir_path, file)
            mod_time = os.path.getmtime(filepath)
            self.file_list.append((mod_time, file))
        sorted_files = sorted(self.file_list, key=lambda x: x[0])                                                   #ulozenie obrazkov do sorted files

        for mod_time, file_name in sorted_files:
            if file_name.endswith('.bmp') or file_name.endswith('.png'):                                            #overenie ci obrazok je nacitany vo formáte bmp alebo png
                file_path = os.path.join(dir_path, file_name)                                                       #vytvorenie celej cesty k obrázku
                img = cv2.imread(file_path)                                                                         #nacitanie obrazku pomocou opencv
                img_flip = cv2.flip(img, 0)
                img_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img_grey = cv2.flip(img_grey, 0)
                gray = cv2.cvtColor(img_flip, cv2.COLOR_BGR2GRAY)
                blur = cv2.bilateralFilter(gray, 8, 80, 80)                                                         #vytvotrenie blurovanej fotky
                ret = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)[1]                                       #vytvotrenie masky
                self.counter_image += 1
                contour, _ = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                      #nájdenie vsetkých kontúr na obrázku
                _, mask = cv2.threshold(img_grey, 124, 255, cv2.THRESH_BINARY_INV)

                for c in contour:                                                                                   #for cyklus ktorý prechádza kazdú kontúru
                    if len(c) >= 5:
                        ellipse = cv2.fitEllipse(c)                                                                 #prisposobenie kazdeho tvaru na elipsu
                        (x, y), (major, minor), angle = ellipse                                                     #parametre elipsy
                        self.major_axes.append(major)
                        minor, major = ellipse[1]

                        if minor <= (self.minor_max * 2) and minor >= (self.minor_min * 2):                         #podmienka, kde sa vykreslia všetky korektne stopy zelenou farbou
                            cv2.ellipse(img_flip, ellipse, (0, 255, 0), 2)                                        #vyznačenie elipsy na obrázku
                            file_output = f"{dir_path}_output"
                            if not os.path.exists(file_output):
                                os.makedirs(file_output)
                            output_filename = f"{file_output}/processed_image_{file_name}.jpg"                  #ulozenie obrázka s vyznacenými elipsami

                            cv2.imwrite(output_filename, img_flip)

                            mask = np.zeros_like(img_grey)
                            cv2.ellipse(mask, ellipse, 255, -1)
                            min_val, max_val, _, _ = cv2.minMaxLoc(img_grey, mask=mask)                      #vypocet odtiena sedej
                            self.grey_append.append(max_val)
                            area = np.pi / 4 * major / 2 * minor / 2                                        #vypocet plochy daneho tvaru
                            self.ellipse_coordinates_area.append(area)
                            formatted_number = "{:.2f}".format(minor/2)
                            formatted_number2 = "{:.2f}".format(area)
                            print("Vedlajsia os:", formatted_number)
                            print("Odtien sivej:", max_val)
                            print("Plocha:",formatted_number2)

                            if major < minor:
                                self.minor_axes.append(major / 2)
                            else:
                                self.minor_axes.append(minor / 2)
                            self.counter += 1

                            if self.counter_image <= 10:                                                      #podmienky posunu x a y súradnice podľa toho ako sa načítajú obrázky
                                y = y / 2 + 1024 * (self.counter_image - 1)
                                x = x / 2
                            elif self.counter_image <= 20:
                                y = y / 2 + 1024 * (self.counter_image - 11)
                                x = x / 2 + 1280
                            elif self.counter_image <= 30:
                                y = y / 2 + 1024 * (self.counter_image - 21)
                                x = x / 2 + 1280 * 2
                            elif self.counter_image <= 40:
                                y = y / 2 + 1024 * (self.counter_image - 31)
                                x = x / 2 + 1280 * 3
                            elif self.counter_image <= 50:
                                y = y / 2 + 1024 * (self.counter_image - 41)
                                x = x / 2 + 1280 * 4
                            elif self.counter_image <= 60:
                                y = y / 2 + 1024 * (self.counter_image - 51)
                                x = x / 2 + 1280 * 5
                            elif self.counter_image <= 70:
                                y = y / 2 + 1024 * (self.counter_image - 61)
                                x = x / 2 + 1280 * 6
                            else:
                                y = y / 2 + 1024 * (self.counter_image - 71)
                                x = x / 2 + 1280 * 7

                            self.y_append.append(y)
                            self.x_append.append(x)
                            self.minor_append.append(minor / 2)
                            self.major_append.append(major / 2)
                            self.angle_append.append(angle)
                        elif minor > (self.minor_max * 2) and minor < (self.minor_min * 2) or major >= 0 and minor >= 0:         # else podmienka, kde sa vykreslia všetky zamietnuté cervenou farbou
                            cv2.ellipse(img_flip, ellipse, (0, 0, 255), 2)

        self.counter_p.append(self.counter)
        mean = statistics.mean(self.minor_append)
        self.mean_n.append(mean)
        plt.hist(self.minor_axes, bins=130, histtype='step', fill=True)                                         #vykreslenie histogramu
        plt.xlim([4, 40])
        plt.xlabel('Velkost vedlajsej osi')
        plt.ylabel('Pocet objektov')
        plt.savefig(f'histogram_{dir_path}.png')
        plt.show()

        with open(f'ellipse_coordinates_{dir_path}.csv', 'w', newline='') as csvfile:                                 #vytvorenie csv suboru zo vsetkych najdenych parametrov
            writer = csv.writer(csvfile)
            writer.writerow(self.header)
            for value in range(len(self.x_append)):
                writer.writerow([self.x_append[value], self.y_append[value], self.minor_append[value],
                                 self.major_append[value], self.angle_append[value],
                                 self.ellipse_coordinates_area[value], self.grey_append[value]])
            writer.writerow(self.mean_name + self.mean_n)
            writer.writerow(self.counter_name + self.counter_p)


def main():                                                      #main kde volám funkciu Analyzer s parametrami minor_min a minor _max
    parameter = input("Enter a parameter (ALFA or NEUT): ")
    if parameter == "ALFA":
        minor_min = 3
        minor_max = 30
    elif parameter == "NEUT":
        minor_min = 4
        minor_max = 40
    else:
        print("Invalid parameter!")
        return
    dir_path = "Img0002535"                  #cesta k súboru z ktoreho sa obrazky nacitaju
    analyzer = ImageAnalyzer(minor_min, minor_max)
    analyzer.analyze_images(dir_path)


if __name__ == '__main__':
    main()                                #zavolanie funkcie main
