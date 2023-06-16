#Analiza ruchu kropli (z serii zdjęć) - wyznaczenie rozciągnięcia kropli w czasie (w pionie i poziomie), wizualizacja i serializacja wyników

import cv2
import time
import msgpack
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

path = "C:/Users/Karol/Downloads/projekt_python_automatyzacja/15mm5Hz_C001H001S0001"

dxarr = []
dyarr = []
tarr = []
ytopcrop = 300
ybotcrop = 660

start_time = time.time()

# Wczytywanie zdjec
files = [f for f in listdir(path) if isfile(join(path, f))]
search_photo = lambda a, b: [a + '/'+ f for f in listdir(a) if f.endswith(b)]
fextension = '.tif'
files = search_photo(path, fextension)
print('ilosc zdjec:', len(files))

# Analiza zdjec
for i in range (0, len(files)):
    img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
    kropla = img[ytopcrop:ybotcrop,0:1024]
    blur = cv2.GaussianBlur(kropla, (5,5), 0)
    ret3,th3 = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cont_th3 = cv2.bitwise_not(th3)
    cont_th3 = cont_th3.astype(np.uint8)

    # wyznaczanie konturu
    contours, hierarchy = cv2.findContours(cont_th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    target = max(contours, key=lambda x: cv2.contourArea(x))
    
    x, y, w, h = cv2.boundingRect(target)
    cv2.rectangle(kropla, (x, y), (x+w,y+h), 0, 5)

    dxarr.append(w)
    dyarr.append(h)
    tarr.append(i)

# Serializacja danych
data = [dxarr, dyarr]
serialized_data = msgpack.packb(data)

with open('serialized_data.msgpack', 'wb') as file:
    file.write(serialized_data)

with open('serialized_data.msgpack', 'rb') as file:
    data_file = file.read()

unpacked_serialized_data = msgpack.unpackb(data_file)

# Mierzenie czasu wykonywania programu
print("czas wykonywania programu")
print("%s seconds" % (time.time() - start_time))

# Rysowanie wykresu
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_ylabel('zmiana szerokosci', color=color)
#ax1.plot(tarr, dxarr, color=color)
ax1.plot(tarr, unpacked_serialized_data[0], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('zmiana wysokosci', color=color)
#ax2.plot(tarr, dyarr, color=color)
ax2.plot(tarr, unpacked_serialized_data[1], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout() 
plt.show()

# Analiza pojedynczej kropli
images = [img, 0, th3,
          kropla, 0, cont_th3]
titles = ['Original Image','Histogram','Thresholding',
          'Cropped Image','Histogram',"Thresholding negative"]
for i in range(2):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()



