# ImageAnalyzer

ImageAnalyzer, je program, ktorý na snímkach hľadá tvary, nájde ich parametre a označí dané tvary na snímke.
Predtým ako sa program spustí je potrebné nainštalovať python 3.9 a OpenCV knižnicu pomoco:
```bash
pip install opencv-python
```
Týmto krokom sa bude dať spustiť program ImageAnalyzer.

## Opis kódu
Na začiatku kódu sú definované všetky premenné, ktoré v kóde vystupú, polia, počítadlá atď.

Ďalšou časťou kódu je načítanie sady obrázkov. Tieto obrázky sú zo súboru načítavané podľa dátumu ich vytvorenia.
```python
dir_path = "Img0002535"            
    images = os.listdir(dir_path)  
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = [file for file in images if os.path.splitext(file)[-1].lower() in image_extensions]

    for file in image_files:                                
        filepath = os.path.join(dir_path, file)
        mod_time = os.path.getmtime(filepath)
        file_list.append((mod_time, file))
    sorted_files = sorted(file_list, key=lambda x: x[0]) 
```
Potom v poli sorted_files sa nachádzajú zoradené obrázky, ktoré pomocou for cyklu iterujeme po jednom
```python
for mod_time, file_name in sorted_files:
```
Pomocou funkcie z opencv dokážeme načítať každý obrázok.
```python
file_path = os.path.join(dir_path, file_name) 
img = cv2.imread(file_path)
```
Na tento obrázok sa aplikujú filtre, maska a nájdu sa kontúry.
```python
gray = cv2.cvtColor(img_flip, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray, 8, 80, 80)                
ret = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)[1]
contour, _ = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
Po nájdení kontúr sa pomocou for cyklu prejde každá kontúra, ktorú knižnica OpenCV našla. Na každú kontúru sa použije funkcia fitEllipse(), ktorá prispôsobí danú kontúru elipse. Ďalším krokom je z tejto elipsy získať parametre.
```python
for c in contour: 
if len(c) >= 5:
ellipse = cv2.fitEllipse(c)      #prisposobenie kazdeho tvaru na elipsu
(x, y), (major, minor), (angle) = ellipse #parametre elipsy
```
Pomocou podmienky, ktorá je zároveň zamietacím kritériom v kóde, je potrebné dané nájdené elipsy vykresliť na obrázok zelenou farbou aby sme ich označili ako korektné a zároveň uložiť ich parametre a vypočítať plochu a odtieň sivej. Vytvárame súbor ktorý má názov načítaného priečinku_output čo nám určuje výsledný priečinok v ktorom sa nachádzajú obrázky z vykreslenými elipsami. Obrázok sa uloží pomomcou cv2.imwrite. 
```python
if minor <= (minor_max*2) and minor >=(minor_min*2): 
cv2.ellipse(img_flip, ellipse, (0, 255, 0), 2)
file_output = f"{dir_path}_output"
if not os.path.exists(file_output):
    os.makedirs(file_output)
output_filename = f"{file_output}/processed_image_{file_name}.jpg"
cv2.imwrite(output_filename, img_flip)
```
```python
mask = np.zeros_like(img_grey)
cv2.ellipse(mask, ellipse, 255, -1)   #vypocet odtiena sedej
min_val, max_val, _, _ = cv2.minMaxLoc(img_grey, mask=mask)
grey_append.append(max_val)
area = np.pi / 4 * major / 2 * minor / 2    #vypocet plochy daneho tvaru
ellipse_coordinates_area.append(area)
```

V kóde sa ďalej nachádza, posun x a y súradnice. 
```python
if counter_image <= 10:
    y = y / 2 + 1024 * (counter_image - 1)
    x = x / 2
elif counter_image <= 20:
    y = y / 2 + 1024 * (counter_image - 11)
    x = x / 2 + 1280
.
.
.
.                            
```
V kóde sa nachádza aj kritérium ktoré zamietne dané stopy. Toto kritérium je nastavené ak je hodnota minor menšia alebo väčšia ako daný rozsah. Tieto stopy sa vykreslia červenou farbou.
```python
elif minor > (minor_max * 2) and minor < (minor_min * 2) or major >= 0 and minor >= 0:  # else podmienka, kde sa vykreslia všetky zamietnuté cervenou farbou
cv2.ellipse(img_flip, ellipse, (0, 0, 255), 2)                   
```
Ďalšou časťou kódu je vykreslenie histogramu, pomeru vedľajšej osi k počtu daných objektov. Dajú sa upraviť jeho stĺpce a rovnako aj hodnoty osi x. Následne sa histogram uloží do adresára súboru.
```python
plt.hist(minor_axes, bins=130, histtype='step', fill=True)      #vykreslenie histogramu
plt.xlim([3, 30])
plt.xlabel('Velkost vedlajsej osi')
plt.ylabel('Pocet objektov')
plt.savefig(f'histogram_{dir_path}.png')
plt.show()                            
```

Poslednou časťou kódu je vytvorenie CSV súboru v ktorom sú uložené parametre daných tvarov. Súbor sa uloží do adresára programu.
```python
with open(f'ellipse_cordinates_{dir_path}', 'w',  newline='') as csvfile:  #vytvorenie csv suboru zo vsetkych najdenych parametrov
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for value in range(len(x_append)):
      writer.writerow([x_append[value],y_append[value],minor_append[value],major_append[value],
      angle_append[value],ellipse_coordinates_area[value],grey_append[value]])
    writer.writerow(mean_name + mean_n)
    writer.writerow(counter_name + counter_p)                    
```

Kód má main funkciu v ktorej sú definované minor_min a minor_max, ktoré si užívateľ zvolí podľa toho o aké typy obrázkov sa jedná. Ak sa vyhodnocujú alfa častice, parametere sa nastavia na 3 a 30 ak neutróny tak 4 a 40. Tieto hodnoty nám hovoria o rozsahu daných parametrov. Náslkedne zavoláme metódu Analyzer, ktorá nám spustí daný kód.
```python
def main():                         #main kde volám funkciu Analyzer s parametrami minor_min a minor _max
    minor_min = 4
    minor_max = 40
    Analyzer(minor_min, minor_max)                      
```
