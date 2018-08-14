import socket                   # Import socket module

s = socket.socket()             # Create a socket object
host = socket.gethostname()     # Get local machine name
port = 60000                    # Reserve a port for your service.

s.connect((host, port))
b = bytearray()
while True:
    print('receiving data...')
    data = s.recv(1024)
    b.append(data)
    if not data:
        break
print(b)
print('Successfully get the file')
s.close()
print('connection closed')




import cython

cpdef unsigned char[:, :] threshold_fast(int T, unsigned char [:, :] image):
 05:     # set the variable extension types
 06:     cdef int x, y, w, h
 07: 
 08:     # grab the image dimensions
+09:     h = image.shape[0]
+10:     w = image.shape[1]
 11: 
 12:     # loop over the image
+13:     for y in range(0, h):
+14:         for x in range(0, w):
 15:             # threshold the pixel
+16:             image[y, x] = 255 if image[y, x] >= T else 0
 17: 
 18:     # return the thresholded image
+19:     return image





##
##
##
##import os
##import io
##from PIL import Image
##import cv2
##
##pipe = os.open("fifoTest", os.O_RDONLY, os.O_NONBLOCK)
##
##imgBytes = os.read(pipe, 117966)
##img = Image.open(io.BytesIO(imgBytes))
##print(imgBytes)
##print(len(imgBytes))
###cv2.imshow("img", img)
##
##input("there?")
##    
               
