from scipy.spatial import distance as dist
from collections import OrderedDict, deque, Counter
import numpy as np
import pprint

class Trackable():
    def __init__(self, id, initialLocation, traceLength=20, life=10):
        self.id = id
        self.features = dict()
        self.trace = deque(maxlen=traceLength)
        self.trace.appendleft(initialLocation)
        
        self.features["currentLocation"] = initialLocation
        self.features["life"] = life

class Tracker():
    def __init__(self, life=10):
        self.nextID = 0
        self.objects = OrderedDict()
        self.trackLife = life

    def Register(self, object):
        self.objects[self.nextID] = object
        self.nextID += 1

    def DeRegister(self, obj):
        print("deleting: ", end="")
        print(obj.id)
        del self.objects[int(obj.id)]
        
    def Update(self):
        print("you need to implement this method!")


class CentroidTracker(Tracker):
    def __init__(self, life=10):
        Tracker.__init__(self)
        
    def Update(self, rects=None):
        print("overrides base class method.")
        if(rects is None):
            for key in list(self.objects.keys()):
                obj = self.objects.get(key)
                obj.features["life"] -= 1
                if(obj.features["life"] == 0):
                    self.DeRegister(obj)
                    
            #return self.objects
            return
        
        centroids = np.zeros((len(rects), 2), dtype="int")
        
        for(i, (x0, y0, x1, y1)) in enumerate(rects):
            cX = int((x0 + x1) / 2.0)
            cY = int((y0 + y1) / 2.0)
            centroids[i] = (cX, cY)
            
        if(len(self.objects) == 0):
            for i in range(0, len(centroids)):
                self.register(centroids[i])
                
        else:
            print("here")
            objCentroids = []
            objIDs = list(self.objects.keys())
            for obj in list(ct.objects.values()):#.features["currentLocation"]):
                objCentroids.append(obj.features["currentLocation"])


def Test2(ct):
    ct.Register(Trackable(str(ct.nextID), (1,1)))
    ct.Register(Trackable(str(ct.nextID), (2,2)))
    ct.Register(Trackable(str(ct.nextID), (3,3)))
    ct.Register(Trackable(str(ct.nextID), (4,4)))
    ct.Register(Trackable(str(ct.nextID), (5,5)))
    ct.Register(Trackable(str(ct.nextID), (6,6)))
    

def Test1(ct):
    ct.Register(Trackable(str(ct.nextID), (0,0)))
    print(list(ct.objects.keys()))
    #print(ct.objects["0"].features["life"])
    ct.Update([(1, 2, 1, 2)])
    ct.Update()
    ct.Update()
    ct.Register(Trackable(str(ct.nextID), (1,1)))
    ct.Update([(3,3, 3, 3)])
    print(list(ct.objects))
    ct.Update()
    ct.Update()
    print(len(ct.objects.keys()))
    print(list(ct.objects))
    
    for i in range(0, 6):
        ct.Update()
        print(len(ct.objects.keys()))
        print(list(ct.objects))
    
    ct.Update()
    print(len(ct.objects.keys()))
    print(list(ct.objects))
    
    ct.Update()
    print(len(ct.objects.keys()))
    print(list(ct.objects))


def Test3(ct):
    ct.Register(Trackable(ct.nextID, (0,0)))
    print(ct.objects[0].features["life"])
    print(list(ct.objects))
    ct.Update()


if(__name__ == "__main__"):
    ct = CentroidTracker()
    Test2(ct)
    for loc in list(ct.objects.values()):#.features["currentLocation"]):
        print(loc.features["currentLocation"])
    Test1(ct)





















    