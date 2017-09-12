
#***********************************************************************************
#***********************************************************************************
class DLink(object):
    '''
***********************************************************************************
LINK CLASS

    This class creates directed links between the node in which it belongs to and
    a neighbor node.  

    Properties:

        neighbor

        weight

    Functions:

        ToString(self)

        Help()
***********************************************************************************
'''
    def __init__(self, neighbor, weight):
        self.neighbor = neighbor
        self.weight = weight

    def ToString(self):
        return str(self.neighbor.name) + " " + str(self.weight)

    def Help():
        print(Node.__doc__)
        return



#***********************************************************************************
#***********************************************************************************
class Node(object):
    '''
***********************************************************************************
NODE CLASS

    This class is to be used in conjunction with the Link Class for a 
    viariety of tree or graph based applications. Not all fields are mandatory.

    Properties:
    
        name: the name of the node
        
        key: the lowest value has the highest priority.  For search algorithms
            this is usually the total distance from start to said node

        parent: the parent node.  usually set within a search algorithm or
            tree building algorithm

        color: can be either 0, 1, or 2 to track if a node is either not visited,
            visited but not all children yet, and visited and all children visited
            (this is the mechanism to handle cycles as implemented here)

        discoveredTime and finishTime: useful for tracking the number of nodes
            visited since start or beyond (which itself has other uses)

        estimatedDistance: heuristic value which gives an estimated "distance"
            from said node to a goal node

        links[]: list of directed links.  for an undirected graph see Graph.Help()

    Functions:

        CompareTo(self, other)

        HasLinks(self)

        ToString(self)

        Help() --> Prints the __doc__ (i.e. these comments)

***********************************************************************************        
'''
    def __init__(self, name, key):
        self.name = name
        self.key = key
        self.parent = ""
        self.color = 0
        #self.discoveredTime = 0
        #self.finishTime = 0
        self.h = 0
        self.links = []

    '''Compares the key value of the nodes'''
    def CompareTo(self, other):
        if(self.key < other.key):
            return -1
        if(self.key > other.key):
            return 1
        else:
            return 0

    '''Checks if the node has outbound neighbors'''
    def HasLinks(self):
        if (len(self.links) > 0):
            return True
        else:
            return False

    def ToString(self):
        return "< " + self.name + " - " + str(self.key) + " >"

    def Help():
        print(Node.__doc__)
        return



#***********************************************************************************
#***********************************************************************************
class Stack(object):
    '''
***********************************************************************************
STACK CLASS

    This is a standard last-in-first-out stack.

    Properties:

        nodes[]

    Functions:

        Push(self, node)

        Pop(self, node)

        IsEmpty(self)

        Count(self)

        NextNode(self)

        PrintStack(self)

        Help()

'''
    def __init__(self):
        self.stack = []

    def __getitem__(self, key):
        return self.stack[key]

    def Push(self, node):
        self.stack.append(node)

    def Pop(self):
        if(self.IsEmpty()):
            print("Stack is empty!")
            return
        else:       
            node = self.stack[len(self.stack) - 1]
            del self.stack[len(self.stack) - 1]
            return node

    def IsEmpty(self):
        if(len(self.stack) == 0):
            return True
        else:
            return False

    def Count(self):
        return len(self.stack)

    def NextNode(self):
        return self.stack[len(self.stack) - 1]

    def PrintStack(self):
        if(len(self.stack) > 0):
            for node in self.stack:
                print(node.ToString())
        else:
            print("the stack is empty!")

    def Help():
        print(Stack.__doc__)


#***********************************************************************************
#***********************************************************************************
class FifoQueue(object):
    '''
***********************************************************************************
FIFO QUEUE CLASS

    This is a standard first-in-first-out queue.

    Properties:

        nodes[]

        head

        tail

    Functions:

        EnQueue(self, node)

        DeQueue(self)

        RemoveTail(self)

        IsEmpty(self)

        Count(self)

        NextNode(self)

        LastNode(self)

        PrintQueue(self)

        Help()

***********************************************************************************
'''
    def __init__(self):
        self.queue = []
        #self.head = ""
        #self.tail = ""
        
    def __getitem__(self, key):
        return self.queue[key]

    def EnQueue(self, node):
        self.queue.append(node)

    def DeQueue(self):
        if(self.IsEmpty()):
            print("Queue is empty!!")
            return
        else:
            node = self.queue[0]
            for i in range(1, len(self.queue)):
                self.queue[i - 1] = self.queue[i]
            del self.queue[len(self.queue) - 1]
            return node

    def RemoveTail(self):
        if(self.IsEmpty()):
            print("Queue is empty!")
            return
        else:       
            node = self.queue[len(self.queue) - 1]
            del self.queue[len(self.queue) - 1]
            return node

    def IsEmpty(self):
        if(len(self.queue) == 0):
            return True
        else:
            return False

    def Count(self):
        return len(self.queue)

    def PrintQueue(self):
        if(len(self.queue) > 0):
            for node in self.queue:
                print(node.ToString())
        else:
            print("the queue is empty!")

    def NextNode(self):
        if(self.IsEmpty()):
            print("Queue is empty!")
            return
        else:
            return self.queue[0]

    def LastNode(self):
        if(self.IsEmpty()):
            print("Queue is empty!")
            return
        else:
            return self.queue[len(self.queue) - 1]

    def Help():
        print(FifoQueue.__doc__)



#***********************************************************************************
#***********************************************************************************
class MinHeap(object):
    '''
***********************************************************************************  
MINIMUM HEAP CLASS

    This class builds a heap with the lowest key always on top and typically
    used as a priority queue (although has many uses).

    Properties:

        heap[]

    Functions:

        IsEmpty(self)

        Count(self)

        Peek(self)

        RemoveLast(self)

        Insert(self, node) [enQueue]

        ExtractMin(self) [deQueue]

        MinHeapify(self, i)

        GetNode(self, i)

        GetNodeIndex(self, node)

        RemoveNode(self, node):

        RemoveNodeAt(self, i):

        PrintHeap(self):

        Help()
        
***********************************************************************************         
'''
    def __init__(self):
        self.heap = []

    def __getitem__(self, key):
        return self.heap[key]

    def IsEmpty(self):
        if(len(self.heap) == 0):
            return True
        else:
            return False

    def Count(self):
        return len(self.heap)

    def Peek(self):
        if(len(self.heap) > 0):
            return self.heap[0]
        else:
            print("The heap is empty.")
            return Node("", -1)

    def RemoveLast(self):
        i = len(self.heap) - 1
        if(i < 0):
            return
        else:
            return self.heap.pop()

    def Insert(self, node):
        self.heap.append(node)
        i = len(self.heap) - 1
        while(i > 0):
            j = int(((i + 1) / 2 ) - 1)
            node = self.heap[j]
            if(node.CompareTo(self.heap[i]) <= 0):
                break
            temp = self.heap[i]
            self.heap[i] = self.heap[j]
            self.heap[j] = temp
            i = j

    def ExtractMin(self):
        if(len(self.heap) <= 0):
            print("heap is empty!!")
            return 
        elif(len(self.heap) == 1):
            return self.RemoveLast()
        else:
            min_ = self.heap[0]
            self.heap[0] = self.RemoveLast()
            self.MinHeapify(0)
            return min_

    '''Maintains the heap property. i is the index to heapify from.'''
    def MinHeapify(self, i):
        smallest = len(self.heap) 
        left = 2 * (i + 1) -1
        right = 2 * (i + 1)
        if(left < len(self.heap) and self.heap[left].CompareTo(self.heap[i]) < 0):
            smallest = left
        else:
            smallest = i
        if(right < len(self.heap) and self.heap[right].CompareTo(self.heap[smallest]) < 0):
            smallest = right
        if(smallest != i):
            temp = self.heap[i]
            self.heap[i] = self.heap[smallest]
            self.heap[smallest] = temp
            self.MinHeapify(smallest)
        
    def GetNode(self, i) -> Node:
        return self.heap[i]

    '''returns the index of the given node.'''
    def GetNodeIndex(self, node):
        for i in range(0, len(self.heap)):
            n = self.heap[i]
            if(n.name == node.name and n.key == node.key):
                return i
        return -1

    def RemoveNode(self, node):
        i = self.GetNodeIndex(node)
        if(i is not -1):
            removed = self.heap[i]
            self.heap[i] = self.heap[len(self.heap) - 1]
            self.RemoveLast()
            self.MinHeapify(i)
            return removed
        else:
            return

    def RemoveNodeAt(self, i):
        pass

        
    def PrintHeap(self):
        if(len(self.heap) > 0):
            for node in self.heap:
                print(node.ToString())
        else:
            print("the heap is empty!")
            
    def Help():
        print(MinHeap.__doc__)
        return
    

#***********************************************************************************
#***********************************************************************************
class Graph(object):
    '''
************************************************************************************
GRAPH CLASS

    The graph can be a directed or undirected representation of a state-space
    problem and can hold various forms such as a BST, MST, DAG, etc...  Duplicate
    nodes are not allowed, however a node can have multiple inbound/outbound arcs.

    Properties:

        nodes

        numNodes

        numLinks

        source

        destination

    Functions:

        Generate(self, filePath)

        Add(self, node)

        Contains(self, nodeName)

        PrintGraph(self)

        GetPath(self, destination)

        PrintPath(self, destination)

        GetNode(self, nodeName)


*************************************************************************************
'''
    def __init__(self):
        self.nodes = []
        self.numNodes = 0
        self.numLinks = 0
        #self.source = object()
        #self.destination = object()

    def __getitem__(self, key):
        return self.nodes[key]

    def Generate(self, filePath):
        pass

    def Add(self, node):
        if(self.Contains(node.name)):
            print("duplicate nodes not allowed. Node not added.")
            return
        else:
            self.nodes.append(node)

    def Contains(self, nodeName):
        for n in self.nodes:
            if(n.name == nodeName):
                return True
        return False

    def PrintGraph(self):
        if(len(self.nodes) > 0):
            for n in self.nodes:
                print(n.name, end='', flush=True)
                if(n.HasLinks):
                    for l in n.links:
                        print(" --" + str(l.weight) + "--> " + l.neighbor.name + " )", end='', flush=True)
                print("")
        else:
            print("Graph is empty")


    def GetPath(self, destination):
        path = []
        node = destination
        path.append(destination)
        while(node.parent is not ""):
            node = node.parent
            path.append(node)
        return list(reversed(path))


    def PrintPath(self, destination):
        path = self.GetPath(destination)
        for n in path:
            print("-" +str(n.key)+  "->(" + n.name + ")", end='', flush=True)
            if(n is destination):
                print("")
        return



'''
*********************************************************************************************
*********************************************************************************************
*********************************************************************************************
*********************************************************************************************


Below are fully functional testing blocks.  Uncomment each block to see the behavior of the
different classes.

'''

'''
#stack testing

myStack = Stack()
print(myStack.IsEmpty())
for i in range(0, 10):
    newNode = Node(("node_%i" %i), 1)
    myStack.Push(newNode)
print(myStack.IsEmpty())
myStack.PrintStack()
print(str(myStack.Count()))
myStack.Pop()
print(str(myStack.Count()))
myStack.PrintStack()
print()
for i in range(myStack.Count()):
    print(myStack.Pop().name)
    print("count: ", myStack.Count())

myStack.PrintStack()
print(myStack.IsEmpty())
myStack.Pop()
'''



#Fifo queue testing

#FifoQueue.Help()
#myQueue = FifoQueue()
#myQueue.LastNode()
#myQueue.NextNode()
#myQueue.DeQueue()
#myQueue.PrintQueue()
#myQueue.RemoveTail()
toCheck = FifoQueue()
checked = FifoQueue()

domA = [1,2,3,4]
domB = [1,2,3,4]
domC = [1,2,3,4]
domD = [1,2,3,4]
domE = [1,2,3,4]

count = 0

a = 0
b = 0
c = 0
d = 0
e = 0

for i in range(0, 11):
    newNode = Node(("node_%i" %i), i)
    toCheck.EnQueue(newNode)


while(toCheck.Count() > 0 and checked.Count() != 11):
    count += 1
    x = toCheck.DeQueue()
    if(x.key == 0): # i.e. b != 3
        domB.remove(3)
        count += 1
        checked.EnQueue(x)
        count += 1
        continue
    if(x.key == 1): # c != 2
        domC.remove(2)
        count += 1
        checked.EnQueue(x)
        count += 1
        continue
    if(x.key == 2): #c < d 1234 1234
        if(domC[len(domC) - 1] <= domD[len(domD) - 1]):
            count += 1
            domC.remove(domC[len(domC) - 1])
        if(domD[0] >= domC[0]):
            count += 1
            domD.remove(domD[0])
        checked.EnQueue(x)
        count +=1
        continue
    if(x.key == 3): # e < a 1234 1234
        if(domA[len(domA) - 1] <= domE[len(domE) - 1]):
            count += 1
            domE.remove(domE[len(domE) - 1])
        if(domA[0] >= domE[0]):
            count += 1
            domA.remove(domA[0])
        checked.EnQueue(x)
        count +=1
        continue
    if(x.key == 4): #e < b
        if(domB[len(domB) - 1] <= domE[len(domE) - 1]):
            count += 1
            domE.remove(domE[len(domE) - 1])
        if(domB[0] >= domE[0]):
            count += 1
            domB.remove(domB[0])
        checked.EnQueue(x)
        count +=1
        continue
    if(x.key == 5): #e < c
        if(domC[len(domC) - 1] <= domE[len(domE) - 1]):
            count += 1
            domE.remove(domE[len(domE) - 1])
        if(domC[0] >= domE[0]):
            count += 1
            domC.remove(domC[0])
        checked.EnQueue(x)
        count +=1
        continue
    if(x.key == 6): #e < d
        if(domD[len(domD) - 1] <= domE[len(domE) - 1]):
            count += 1
            domE.remove(domE[len(domE) - 1])
        if(domD[0] >= domE[0]):
            count += 1
            domD.remove(domD[0])
        checked.EnQueue(x)
        count +=1
        continue
    if(x.key == 7): # a == d
        i = 0
        while(True):
            if(domA[i] not in domD):
                count += 1
                domA.remove(domA[i])
                i -= 1
            i += 1
            if(i == len(domA)):
                break
        i = 0
        while(True):
            if(domD[i] not in domA):
                count += 1
                domD.remove(domD[i])
                i -= 1
            i += 1
            if(i == len(domD)):
                break       

print(domA)
print(domB)
print(domC)
print(domD)
print(domE)
print("count after constraint checking: " + str(count))

for r in domA:
    for s in domB:
        for t in domC:
            for u in domD:
                for v in domE:
                    a = r
                    b = s
                    c = t
                    d = u
                    e = v
                    count += 1
                    if(b == 3 or c == 2 or a == b or b == d or b == c or a != d or e >= a or e >= b or e >= c or e >= d or c >= d):
                        continue
                    else:
                        print("Model Found!")
                        print(a)
                        print(b)
                        print(c)
                        print(d)
                        print(e)

print("count after finding all possible models: " + str(count))
    

#print(myQueue[3].name)

#myQueue.PrintQueue()
#print("length: " + str(myQueue.Count()))
#print("dequeue: " + myQueue.DeQueue().name)
#myQueue.PrintQueue()
#print("length: " + str(myQueue.Count()))

#print("removing tail: " + myQueue.RemoveTail().name)
#print("length: " + str(myQueue.Count()))
#myQueue.PrintQueue()



'''
#graph testing

G = Graph()

for i in range(0, 10):
    newNode = Node(("node_%i" %i), 1)
    G.Add(newNode)

print("before links: ")
G.PrintGraph()
print()
print("after links: ")
G[0].links.append(DLink(G[5], 1))
G[0].links.append(DLink(G[9], 99))
G[2].links.append(DLink(G[4], 1))
G[2].links.append(DLink(G[7], 1))
G[2].links.append(DLink(G[6], 5))
#G[2].links.append(DLink(G[0], 1))
G[4].links.append(DLink(G[1], 1))
G[5].links.append(DLink(G[2], 1))
G[7].links.append(DLink(G[9], 1))
G[6].links.append(DLink(G[1], 1))
G[6].links.append(DLink(G[8], 3))
G[6].links.append(DLink(G[2], 2))
G.PrintGraph()
print()
#print("after MakeUndirected: ")


G[9].parent = G[7]
G[7].parent = G[2]
G[2].parent =  G[5]
G[5].parent = G[0]
path, cost = G.GetPath(G[9])
for n in path:
    print(n.ToString())
print("cost: " + str(cost))
print(G[9].ToString())

G.PrintPath(G[6])
'''



'''
#heap testing

MinHeap.Help()

myHeap = MinHeap()
myHeap.Peek()
print(myHeap.IsEmpty())
for i in range(0, 10):
    newNode = Node(("node_%i" %i), 9-i)
    myHeap.Insert(newNode)
print(myHeap.IsEmpty())
myHeap.PrintHeap()
myHeap[2].links.append(DLink(myHeap[4], 2))
myHeap[4].links.append(DLink(myHeap[1], 5))
myHeap[4].parent = myHeap[2]
print(myHeap[4].parent.ToString())
print(myHeap[2].ToString())
print(myHeap[2].HasLinks())
print(myHeap[3].HasLinks())
print(myHeap.GetNodeIndex(Node("node_3", 6)))
print(myHeap[8].ToString())
print(myHeap.GetNode(8).ToString())

print("")

print(myHeap.Peek().ToString())

for i in range(0, myHeap.Count()):
    minNode = myHeap.ExtractMin()
    print("Extracted: " + minNode.name + ", key = " + str(minNode.key))
    print("new heap:")
    myHeap.PrintHeap()    
    print("")
print(myHeap.IsEmpty())
print(myHeap.Peek().ToString())
'''

               
