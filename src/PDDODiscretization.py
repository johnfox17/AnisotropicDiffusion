import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import solve

class PDDODiscretization:
    def __init__(self, noisyImage, numNodes, coords, dx, dy, dt, \
            deltaX, deltaY, horizon):
        self.rows, self.columns = np.shape(noisyImage)
        self.noisyImage = noisyImage.flatten()
        self.numNodes = numNodes
        self.coords = coords
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.horizon = horizon
        self.deltaX = deltaX
        self.deltaY = deltaY
        self.bVec10 = np.array([0,1,0])
        self.bVec01 = np.array([0,0,1])
        self.diffOrder = 1

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

    def calcGradients(self, family, xXi, yXi):
        noisyImage = self.noisyImage
        dx = self.dx
        dy = self.dy
        bVec10 = self.bVec10
        bVec01 = self.bVec01
        deltaX = self.deltaX
        deltaY = self.deltaY
        deltaMag = np.sqrt(deltaX**2 + deltaY**2)
        diffMat = np.zeros([3,3])
        g10 = []
        g01 = []
        for iFamilyMember in range(len(family)):
            currentFamilyMember = family[iFamilyMember]
            currentXXi = xXi[iFamilyMember]
            currentYXi = yXi[iFamilyMember]
            xiMag = np.sqrt(currentXXi**2+currentYXi**2) 
            pList = np.array([1,currentXXi/deltaMag, currentYXi/deltaMag])
            weight = np.exp(-4*(xiMag/deltaMag)**2)
            diffMat += weight*np.outer(pList,pList)*dx*dy
            
        for iFamilyMember in range(len(family)):
            currentFamilyMember = family[iFamilyMember]
            currentXXi = xXi[iFamilyMember]
            currentYXi = yXi[iFamilyMember]
            xiMag = np.sqrt(currentXXi**2+currentYXi**2)
            pList = np.array([1,currentXXi/deltaMag, currentYXi/deltaMag])
            weight = np.exp(-4*(xiMag/deltaMag)**2)
            g10.append(weight*(np.inner(solve(diffMat,bVec10), pList)))
            g01.append(weight*(np.inner(solve(diffMat,bVec01), pList)))
        gradientX = np.inner(np.array(g10), noisyImage[family])
        gradientY = np.inner(np.array(g01), noisyImage[family])
        return gradientX, gradientY

    def timeIntegrate(self):
        numNodes = self.numNodes
        familyMembers = self.familyMembers
        xXis = self.xXis
        yXis = self.yXis
        rows = self.rows
        columns = self.rows
        gradMatX = np.zeros([rows*columns,1]) 
        gradMatY = np.zeros([rows*columns,1])
        for iNode in range(numNodes):
            family = familyMembers[iNode]
            xXi = xXis[iNode]
            yXi = yXis[iNode]
            gradMatX[iNode], gradMatY[iNode] = self.calcGradients(family, xXi, yXi)
        np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\AnisotropicDiffusion\\data\\gradMatX.csv', gradMatX.reshape((300, 300)), delimiter=",")
        np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\AnisotropicDiffusion\\data\\gradMatY.csv', gradMatY.reshape((300, 300)), delimiter=",")
        print('Here')
    def solve(self):
        self.findFamilyMembers()
        self.calcXis()
        self.timeIntegrate()
        print('Done')
        a = input('').split(" ")[0]

