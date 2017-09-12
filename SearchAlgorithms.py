from DataStructures import *
from typing import *

#*************************************************************************
#*************************************************************************
class Searcher(object):
    '''
*************************************************************************
SEARCHER CLASS

    This clas creates a "searcher" object to carry out searches on a graph.

    Parameters:

        graph: Any state-based data structure with nodes and arcs

        goals[]: The goal nodes for the searching

    Functions:

        Astar(self, source, destination)

        
    
*************************************************************************
'''
    def __init__(self, graph):
        self.graph: Graph = graph
        self.goals = []
        self.goalFound = False

    def DFSHelper(self, source, destination):
        if(self.goalFound == True):
            return
        source.color = 1
       # print("visiting: " + source.name)
        for l in source.links:
          #  print("source: " + source.name + " link: " + l.ToString() + " color: " + str(source.color))
            v = l.neighbor
            if(v.color < 2 and not self.goalFound):
                #print("color: " + v.name)
                v.color = 1
                v.parent = source
                v.key = v.parent.key + l.weight
                if(v == destination):
                   # print("destination: " + v.name)
                    self.goalFound = True
                    return
                else:
                    print("going to: " + v.name)
                    self.DFSHelper(v, destination)
            source.color = 2
        

    def DepthFirstSearchRecursive(self, source, destination):
        for n in self.graph.nodes:
            n.key = 0
            n.color = 0
            n.parent = ""
        self.DFSHelper(source, destination)

    def DepthFirstSearchStack(self, source, destination):
        stack = Stack()
        for n in self.graph.nodes:
            n.color = 0
            n.key = 0
        source.color = 1
        stack.Push(source)
        while(stack.IsEmpty() == False):
            u = stack.Pop()
            print("popping: " + u.name)
            if(u == destination):
                self.goalFound == True
                break
            if(u.HasLinks() and u.color is not 2):
                for l in u.links:
                    v = l.neighbor
                    if (v.color == 0):
                        v.color = 1
                        v.parent = u
                        v.key = u.key + l.weight
                        stack.Push(v)
                        print("pushing: " + v.name)
            u.color = 2                    
        

    def BreadthFirstSearch(self, source, destination):
        FQ = FifoQueue()
        source.color = 1
        source.key = 0
        FQ.EnQueue(source)
        while(FQ.IsEmpty() == False):
            u = FQ.DeQueue()
            if(u.HasLinks()):
                for l in u.links:
                    v = l.neighbor
                    if (v.color == 0):
                        v.color = 1
                        v.parent = u
                        v.key = u.key + l.weight
                        FQ.EnQueue(v)
            u.color = 2


    def LowestCostFirstSearch(self, source, destination):
        for n in self.graph.nodes:
            n.key = 999999999
        source.key = 0
        PQ = MinHeap()
        PQ.Insert(source)
        while(PQ.IsEmpty() == False):
            u = PQ.ExtractMin()
            print("visiting: " + u.name + " with key: " + str(u.key))
            if(u == destination):
                goalFound = True
                break
            if(u.HasLinks()):
                for l in u.links:
                    v = l.neighbor
                    distThruU = u.key +l.weight
                    if(distThruU < v.key):
                        PQ.RemoveNode(v)
                        v.key = distThruU
                        v.parent = u
                        PQ.Insert(v)

    def AStar(self, source, destination):
        for n in self.graph.nodes:
            n.key = 999999999
        source.key = 0
        source.h = 0
        PQ = MinHeap()
        PQ.Insert(source)
        while(PQ.IsEmpty() == False):
            u = PQ.ExtractMin()
            u.key -= u.h
            print("visiting: " + u.name + " with key: " + str(u.key))
            if(u == destination):
                goalFound = True
                break
            if(u.HasLinks()):
                for l in u.links:
                    v = l.neighbor
                    distThruU = u.key +l.weight
                    estDist = self.CalculateHeuristic(v) + distThruU
                    if(estDist < v.key + v.h):
                        PQ.RemoveNode(v)
                        v.key = estDist
                        v.parent = u
                        PQ.Insert(v)


    def CalculateHeuristic(self, node):
        #need to implement this
        return node.h



G : Graph = Graph()

for i in range(0, 14):
    newNode = Node(("node_%i" %i), 1)
    G.Add(newNode)

##G[0].h = 4
##G[1].h = 2
##G[2].h = 1
##G[3].h = 2
##G[4].h = 1
##G[5].h = 0
##G[6].h = 4
##G[7].h = 5
##G[8].h = 2
##G[9].h = 1
##G[10].h = 0
##G[11].h = 0
##G[12].h = 2
##G[13].h = 2
##
##G[0].links.append(DLink(G[1], 4))
##G[0].links.append(DLink(G[2], 5))
##G[0].links.append(DLink(G[3], 1))
##G[1].links.append(DLink(G[4], 1))
##G[1].links.append(DLink(G[5], 3))
##G[2].links.append(DLink(G[6], 7))
##G[2].links.append(DLink(G[7], 5))
##G[3].links.append(DLink(G[8], 4))
##G[3].links.append(DLink(G[9], 1))
##G[3].links.append(DLink(G[10], 3))
##G[7].links.append(DLink(G[11], 5))
##G[7].links.append(DLink(G[12], 1))
##G[8].links.append(DLink(G[12], 3))
##G[9].links.append(DLink(G[13], 1))





G[0].h = 12
G[1].h = 0
G[2].h = 6
G[3].h = 2
G[4].h = 1
G[5].h = 9
G[6].h = 1
G[7].h = 3
G[8].h = 15
G[9].h = 2

print()

G[0].links.append(DLink(G[5], 2))
G[0].links.append(DLink(G[8], 4))
G[2].links.append(DLink(G[4], 15))
G[2].links.append(DLink(G[7], 3))
G[3].links.append(DLink(G[6], 1))
G[3].links.append(DLink(G[9], 7))
G[4].links.append(DLink(G[1], 1))
G[5].links.append(DLink(G[2], 5))
G[6].links.append(DLink(G[1], 1))
G[7].links.append(DLink(G[9], 1))
G[7].links.append(DLink(G[6], 7))
G[8].links.append(DLink(G[3], 13))
G[9].links.append(DLink(G[3], 1))

#toggle this to verify "first" path is found for dfs/bfs
#and to verify the "shortest" path is found for dijkstras/astar/etc
#G[0].links.append(DLink(G[9], 99))

#toggle this to verify cycles are handled in dfs/bfs/etc
#the cycle is G[0] - G[5] - G[2] - G[0]
#G[2].links.append(DLink(G[0], 1))

#G.PrintGraph()
#print()

searcher = Searcher(G)

src = G[0]
dest = G[1]

#searcher.DepthFirstSearchRecursive(src, dest)
#searcher.DepthFirstSearchStack(src, dest)
#searcher.BreadthFirstSearch(src, dest)
#searcher.LowestCostFirstSearch(src, dest)
searcher.AStar(src, dest)
print("finished search!")
print(str(dest.key))
G.PrintPath(dest)









