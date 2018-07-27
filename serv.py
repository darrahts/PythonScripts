import socket
import sys

socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#server = ("52.73.65.98", 1973)

addr = ("localhost", 12345)
socket.bind(addr)

while(True):
    data, addr = socket.recvfrom(1024)
    if(len(data) > 0):
        print(data)
        print(addr)
