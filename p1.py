import socket                   # Import socket module

port = 60000                    # Reserve a port for your service.
s = socket.socket()             # Create a socket object
host = socket.gethostname()     # Get local machine name
s.bind((host, port))            # Bind to the port
s.listen(5)                     # Now wait for client connection.

print ('Server listening....')

while True:
    conn, addr = s.accept()     # Establish connection with client.
    print ('Got connection from', addr)

    filename='Lenna.png'
    f = open(filename,'rb')
    l = f.read(1024)
    while (l):
       conn.send(l)
       l = f.read(1024)
    f.close()

    print('Done sending')
    conn.send('Thank you for connecting')
    conn.close()







##
##import os
##import io
##import cv2
##
##
##
##fifo_name = "fifoTest"
##
##def Test():
##    img = cv2.imread("Lenna.png")
##    data = bytearray(img)
##    try:
##        os.mkfifo(fifo_name)
##        print("made fifo")
##    except FileExistsError:
##        print("fifo exists!")
##    with open(fifo_name, "wb", os.O_NONBLOCK) as f:
##        f.write(data)
##        f.close()
##    print("done!")
####
####
####
####
####
##if (__name__ == "__main__"):
##    Test()
##
