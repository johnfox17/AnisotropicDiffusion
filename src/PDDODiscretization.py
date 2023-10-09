import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import solve
import math

class PDDODiscretization:
    def __init__(self, noisyImage, numNodes, coords, dx, dy, dt, \
            deltaX, deltaY, deltaX2, deltaY2, horizon):
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
        self.deltaX2 = deltaX2
        self.deltaY2 = deltaY2
        self.bVec10 = np.array([0,1,0,0,0,0])
        self.bVec01 = np.array([0,0,1,0,0,0])
        self.bVec20 = np.array([0,0,0,2,0,0])
        self.bVec02 = np.array([0,0,0,0,2,0])
        self.diffOrder = 1

    def findFamilyMembers(self):
        coords = self.coords
        numNodes = self.numNodes
        deltaX = self.deltaX
        deltaX2 = self.deltaX2
        tree = KDTree(coords, leaf_size=2)
        familyMembers = tree.query_radius(coords, r = deltaX)
        familyMembers2 = tree.query_radius(coords, r = deltaX2)
        self.familyMembers = familyMembers
        self.familyMembers2 = familyMembers2

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

    def calcDerivatives(self, family, xXi, yXi):
        noisyImage = self.noisyImage
        dx = self.dx
        dy = self.dy
        bVec10 = self.bVec10
        bVec01 = self.bVec01
        bVec20 = self.bVec20
        bVec02 = self.bVec02
        deltaX = self.deltaX
        deltaY = self.deltaY
        deltaMag = np.sqrt(deltaX**2 + deltaY**2)
        diffMat = np.zeros([6,6])
        g10 = []
        g01 = []
        g20 = []
        g02 = []
        for iFamilyMember in range(len(family)):
            currentFamilyMember = family[iFamilyMember]
            currentXXi = xXi[iFamilyMember]
            currentYXi = yXi[iFamilyMember]
            xiMag = np.sqrt(currentXXi**2+currentYXi**2)
            pList = np.array([1,currentXXi/deltaMag, currentYXi/deltaMag,\
                    (currentXXi/deltaMag)**2, (currentYXi/deltaMag)**2,\
                    (currentXXi/deltaMag)*(currentYXi/deltaMag)])
            weight = np.exp(-4*(xiMag/deltaMag)**2)
            diffMat += weight*np.outer(pList,pList)*dx*dy

        for iFamilyMember in range(len(family)):
            currentFamilyMember = family[iFamilyMember]
            currentXXi = xXi[iFamilyMember]
            currentYXi = yXi[iFamilyMember]
            xiMag = np.sqrt(currentXXi**2+currentYXi**2)
            pList = np.array([1,currentXXi/deltaMag, currentYXi/deltaMag,\
                    (currentXXi/deltaMag)**2, (currentYXi/deltaMag)**2,\
                    (currentXXi/deltaMag)*(currentYXi/deltaMag)])
            weight = np.exp(-4*(xiMag/deltaMag)**2)
            g10.append(weight*(np.inner(solve(diffMat,bVec10), pList)))
            g01.append(weight*(np.inner(solve(diffMat,bVec01), pList)))
            g20.append(weight*(np.inner(solve(diffMat,bVec20), pList)))
            g02.append(weight*(np.inner(solve(diffMat,bVec02), pList)))
        
        gradientX = np.inner(np.array(g10)/np.linalg.norm(np.array(g10)), noisyImage[family])
        gradientY = np.inner(np.array(g01)/np.linalg.norm(np.array(g01)), noisyImage[family])
        #gradientX = np.inner(np.array(g10), noisyImage[family])
        #gradientY = np.inner(np.array(g01), noisyImage[family])

        #laplacian = np.inner((np.array(g20) + np.array(g02))/np.linalg.norm(np.array(g20) + np.array(g02)), noisyImage[family])
        laplacian = np.inner(np.array(g20) + np.array(g02), noisyImage[family])
        return gradientX, gradientY, laplacian

    
    def calcCoefficients(self, gradientX, gradientY):
        K = 0.8*np.linalg.norm([gradientX, gradientY])
        coeffs =  [math.exp(-((np.abs(gradientX)/K)**2)), math.exp(-((np.abs(gradientY)/K)**2))]
        return math.sqrt(coeffs[0]**2+coeffs[1]**2)
        #return coeffs

    def calcGradOfCoeffs(self, family, xXi, yXi, coeffs):
        dx = self.dx
        dy = self.dy
        bVec10 = self.bVec10[:3]
        bVec01 = self.bVec01[:3]
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
        g10 = np.array(g10)/np.linalg.norm(np.array(g10))
        g10 = g10.reshape((1, len(g10)))
        g01 = np.array(g01)/np.linalg.norm(np.array(g01))
        g01 = g01.reshape((1, len(g01)))

        gradCoeffsX = np.inner(g10, np.transpose(coeffs[family]))
        gradCoeffsY = np.inner(g01, np.transpose(coeffs[family]))
        return gradCoeffsX, gradCoeffsY

    def timeIntegrate(self):
        numNodes = self.numNodes
        familyMembers = self.familyMembers
        familyMembers2 = self.familyMembers2 #Used for solving differential equation on neighboring points
        xXis = self.xXis
        yXis = self.yXis
        rows = self.rows
        columns = self.rows
        gradMatX = np.zeros([rows*columns,1]) 
        gradMatY = np.zeros([rows*columns,1])
        coeffs = np.zeros([rows*columns,1])
        gradCoeffsX =  np.zeros([rows*columns,1])
        gradCoeffsY =  np.zeros([rows*columns,1])
        noisyImage = self.noisyImage
        lambd = 0.01
        Laplacian = np.zeros([rows*columns,1])

        for i in range(2):
            for iNode in range(numNodes):
                family = familyMembers[iNode]
                xXi = xXis[iNode]
                yXi = yXis[iNode]
                gradMatX[iNode], gradMatY[iNode], Laplacian[iNode] = self.calcDerivatives(family, xXi, yXi) 
                coeffs[iNode] = self.calcCoefficients(gradMatX[iNode], gradMatY[iNode])
            for iNode in range(numNodes):
                family = familyMembers[iNode]
                xXi = xXis[iNode]
                yXi = yXis[iNode]
                gradCoeffsX[iNode], gradCoeffsY[iNode] = self.calcGradOfCoeffs(family, xXi, yXi, coeffs)
            for iNode in range(numNodes):
                family = familyMembers2[iNode]
                #noisyImage[iNode] = noisyImage[iNode] + lambd*(np.inner(gradCoeffsX[family].flatten(),\
                #        gradMatX[family].flatten()) + np.inner(gradCoeffsY[family].flatten(),gradMatY[family].flatten()))
                noisyImage[iNode] = noisyImage[iNode] + lambd*(np.inner(coeffs[family].flatten(),\
                    gradMatX[family].flatten()) + np.inner(coeffs[family].flatten(),gradMatY[family].flatten()))
            
        self.noisyImage = noisyImage
    
    def solve(self):
        self.findFamilyMembers()
        self.calcXis()
        self.timeIntegrate()
        deNoisedImage = self.noisyImage
        np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\AnisotropicDiffusion\\data\\deNoisedImagePDDO.csv', deNoisedImage.reshape((300, 300)), delimiter=",")
        print('Done PDDO')
        #a = input('').split(" ")[0]

