import sys
sys.path.append("/home/tdarrah/")
import socket
            
            
            
socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server = ("127.0.0.1", 6789)

while(True):
    x = input(": ")
    socket.sendto(str.encode(x), server)
    if(x == "q"):
        break