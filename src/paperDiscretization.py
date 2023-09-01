import numpy as np


class paperDiscretization:
    def __init__(self, noisyImage, coords):
        self.noisyImage = noisyImage
        self.coords = coords
        self.rows, self.columns = noisyImage.shape

    
    def extractPixelMasks(self):
        noiseImage = self.noisyImage
        rows = self.rows
        columns = self.columns
        pixelMasks = []
        for iRow in range(1,rows-1):
            #currentPixelMask = []
            for iCol in range(1,columns-1):
                pixelMasks.append([noiseImage[iRow,iCol], noiseImage[iRow-1,iCol],\
                        noiseImage[iRow+1,iCol],noiseImage[iRow,iCol+1],noiseImage[iRow,iCol-1]])
        self.pixelMasks = pixelMasks

    def calculateGradientOfPixel(self):
        pixelMasks = self.pixelMasks
        gradients = []
        for iIntensities in pixelMasks:
            gradients.append([iIntensities[1]-iIntensities[0], iIntensities[2]-iIntensities[0], \
                    iIntensities[3]-iIntensities[0], iIntensities[4]-iIntensities[0]])
        self.gradients = gradients

    def calcCoefficients(self):
        gradients = self.gradients
        K = 2 #TODO: Local contrast of edge
        coefficients = []
        for iGradients in gradients:
            coefficients.append([np.exp(-((np.abs(iGradients[0])/K)**2)), np.exp(-((np.abs(iGradients[1])/K)**2)),\
                    np.exp(-((np.abs(iGradients[2])/K)**2)), np.exp(-((np.abs(iGradients[3])/K)**2))])
        self.coefficients = coefficients
    
    def solve(self):
        self.extractPixelMasks() 
        self.calculateGradientOfPixel()
        self.calcCoefficients()
        
        #a = input('').split(" ")[0]
        print('What up?')
