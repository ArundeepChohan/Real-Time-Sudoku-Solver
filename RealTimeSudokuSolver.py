import cv2
import mss
import numpy
import os
from pprint import pprint
import mouse
from threading import Thread

try:
    from PIL import Image
except ImportError:
    import Image 
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = os.getcwd()+'/Tesseract-OCR/tesseract.exe'
rows,cols = (9, 9) 
def returnDigit(arr,crop_img,i,j):
    custom_oem_psm_config = r'--psm 6'
    text = pytesseract.image_to_string(crop_img,config=custom_oem_psm_config )
    #print(text)
    number = re.search(r'\d',text)
    if number:
        #print('First number found = {}'.format(number.group()))
        #print(type(number.group()))
        arr[i][j]=int(number.group())
    else:
        #print('0')
        arr[i][j]= 0
        
def findNextCellToFill(arr):
    for x in range(rows):
        for y in range(cols):
            if arr[x][y] == 0:
                return x, y
    return -1, -1

def isValid(arr, i, j, e):
    rowOk = all([e != arr[i][x] for x in range(rows)])
    if rowOk:
        columnOk = all([e != arr[x][j] for x in range(cols)])
        if columnOk:
            secTopX, secTopY = 3*(i//3), 3*(j//3)
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3):
                    if arr[x][y] == e:
                        return False
            return True
    return False        
def solveSudoku(arr,i=0, j=0):
    i, j = findNextCellToFill(arr)
    if i == -1:
        return True
    for e in range(1, 10):
        if isValid(arr,i, j, e):
            arr[i][j] = e
            if solveSudoku(arr, i, j):
                return True
            arr[i][j] = 0
    return False

def main():
    x = 0
    y = 0
    w = 300
    h = 300
    arr = [ [0]*rows for i in range(cols)]
    with mss.mss() as sct:
        while True:
             # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            x,y = mouse.get_position()[0],mouse.get_position()[1]
            #print(x,y)
            monitor = {"top": y, "left": x, "width": w, "height": h}

            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor),dtype=numpy.uint8)
            img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
            img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
            img = cv2.GaussianBlur(img, (1, 1), 0)
            threads=[]
            for i in range(rows):
                h1=int((h/rows)*i)
                h2=int((h/rows)*(i+1))
                for j in range(cols):
                    #print(i,j) 
                    w1=int((w/cols)*j)
                    w2=int((w/cols)*(j+1))
                    #print(w1,w2,h1,h2)
                    crop_img=img[h1:h2,w1:w2]
                    process = Thread(target=returnDigit, args=(arr,crop_img,i,j) )
                    process.start()
                    threads.append(process)
            for process in threads:
                process.join()
           
            pprint(arr)
            solveSudoku(arr)
            pprint(arr)
            cv2.imshow("OpenCV/Numpy normal", img)

if __name__ == "__main__":
    main()     



