import numpy as np
from sklearn.neighbors import KDTree

class PDDODiscretization:
    def __init__(self, numNodes, coords, dx, dy, dt, deltaX, deltaY,\
            horizon):
        
        self.numNodes = numNodes
        self.coords = coords
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.horizon = horizon
        self.deltaX = deltaX
        self.deltaY = deltaY

    def findFamilyMembers(self):
        coords = self.coords
        numNodes = self.numNodes
        deltaX = self.deltaX
        tree = KDTree(coords, leaf_size=2)
        familyMembers = tree.query_radius(coords, r = deltaX)
        self.familyMembers = familyMembers

    def calcXis(self):
        coords = self.coords
        numNodes = self.numNodes
        familyMembers = self.familyMembers

        xXis = []
        yXis = []
        for iNode in range(numNodes):
            family = familyMembers[iNode]
            currentXXis = []
            currentYXis = []
            for iFamilyMember in range(len(family)):
                currentXXis.append(coords[family[iFamilyMember]][0] - coords[iNode][0])
                currentYXis.append(coords[family[iFamilyMember]][1] - coords[iNode][1])
            xXis.append(currentXXis)
            yXis.append(currentYXis)
        self.xXis = xXis
        self.yXis = yXis

    def solve(self):
        self.findFamilyMembers()
        self.calcXis()
        for i in range(4): 
            print(self.xXis[i])
            print(self.yXis[i])
            print(self.familyMembers[i])
            a = input('').split(" ")[0]
        print('Done')
        a = input('').split(" ")[0]

