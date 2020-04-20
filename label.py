import numpy as np
import cv2
import pytesseract
from PIL import Image

#image = Image.open("formexample.png")
#width,height = image.size
#image = image.resize((width*1//2 ,height*1//2 ), Image.ANTIALIAS) 
#rgb_im = image.convert('RGB') 
#quality_val = 90
#rgb_im.save('modified.jpg', quality=quality_val)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
large = cv2.imread('formexample.png')
h = large.shape[0]
w = large.shape[1]
h = h*1/2
w = w*1/2
rgb = cv2.resize(large, (int(w), int(h)) )
#rgb = cv2.pyrDown(large)
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
img = large
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel = np.ones((5, 5), np.uint8)
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

# using RETR_EXTERNAL instead of RETR_CCOMP
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#For opencv 3+ comment the previous line and uncomment the following line
#_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

mask = np.zeros(bw.shape, dtype=np.uint8)
count=0
for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > 0.45 and w > 8 and h > 8:
        cv2.rectangle(rgb, (x-1, y-1), (x+w-1, y+h-1), (0, 255, 0), 2)
        new_img = rgb[y:y+h, x:x+w]
        cv2.adaptiveBilateralFilter(new_img,(11,11),50)
        cv2.threshold(new_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones((9,9),np.uint8) 
        cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)
        cv2.Canny(new_img, 100, 200)
        new_img = cv2.resize(new_img, None, fx=5.5, fy=4.5, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(new_img)
        if text!="" and text!=" ":
            count+=1;
            print('field '+str(count)+') '+text)
            print("\n")


cv2.imshow('rects', rgb)
cv2.waitKey(0)