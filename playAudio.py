from multiprocessing import Process, Event, Value, Lock
import pygame
from datetime import datetime as dt
import time
import select
import socket

def PlaySong(song):
    pygame.mixer.music.load(song)
    pygame.mixer.music.play()
    while(pygame.mixer.music.get_busy()):
        pass
    return

def WatchPort(s, lock, sig):
    while True:
        ready2 = select.select([serverSocket], [], [], .1)
        if(ready2[0]):
            data, adr = s.recvfrom(1024)
            msg = data.decode("ascii")
            print(msg)
            if(msg == "clap"):
                with lock:
                    sig.value = 1
                    break
    

UDP_ADR = ""
UDP_PORT = 6789
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverSocket.bind((UDP_ADR, UDP_PORT))
print(serverSocket)

i = 0

L = Lock()
signal = Value("i", 0)

P = Process(target=WatchPort, args=(serverSocket, L, signal ))
P.e = Event()
P.start()

pygame.mixer.init()

now = dt.now()
startTime = dt(2018, 8, 21, 18, 20)

try:
    while(True):
        with L:
            print("signal value: {}".format(signal.value))
            print("counter value: {}".format(i))
            
        if(i == 0 and dt.now() >= startTime):
            print("playing 1")
            PlaySong("song1short.wav")
            print("playing 2")
            PlaySong("song0shortfaded.wav")
            i = 1
        if(signal.value == 1 and i == 1):
            print("playing 3")
            PlaySong("song2short.wav")
            print("played.")
            with lock:
                signal.value == 2
            break
        else:
            print("waiting...")
            time.sleep(2)
except:
    pass

finally:
    P.e.set()
    #server.finished.value = True
    P.join()
    #server.closeConnection()
    print("done")







