


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
        print(pixelMasks)


    def solve(self):
        self.extractPixelMasks() 
        self.calculateGradientOfPixel()

        #a = input('').split(" ")[0]
        print('What up?')
