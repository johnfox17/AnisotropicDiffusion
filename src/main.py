
import paperDiscretization 
import PDDODiscretization
import sys
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Defining constants
l1=1
l2=1
#N = 100
dx = 1/512
dy = 1/512
#dt = 1/512
#deltaX = 0.006015
#deltaY = 0.006015

#t0 = 0
#tf = 5
#numTimeSteps =int(tf/dt)
xCoords = np.linspace(0.5/300,1+0.5/300,300)#create the discrete x and y grids
yCoords = np.linspace(0.5/300,1+0.5/300,300) #create the discrete x and y grids
indexing = 'xy'
xCoords, yCoords = np.meshgrid(xCoords, yCoords, indexing=indexing)
xCoords = xCoords.reshape(-1, 1)
yCoords = yCoords.reshape(-1, 1)
coords = np.array([xCoords[:,0], yCoords[:,0]]).T
numNodes = len(xCoords)
print(np.shape(coords))
#np.savetxt('/home/doctajfox/Documents/Thesis_Research/AnisotropicDiffusion/data/coords.csv', coords, delimiter=",")
#np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\AnisotropicDiffusion\\data\\coords.csv', coords, delimiter=",")
#print('Done')
#a = input('').split(" ")[0]


def main():
    
    if sys.platform.startswith('linux'):
        pathToLena = \
            '/home/doctajfox/Documents/Thesis_Research/AnisotropicDiffusion/data/testImage.jpg'
    else:
        pathToLena = 'C:\\Users\\docta\\Documents\\Thesis\\AnisotropicDiffusion\\data\\testImage.jpg'

    # load image as pixel array
    image = cv2.imread(pathToLena)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # summarize shape of the pixel array
    rows, columns = image.shape
    #Create white gaussian noise
    mean = 0
    std = 20
    numNodes = rows*columns
    noise = np.round(np.abs(np.random.normal(mean, std, size=numNodes).reshape((rows, columns))))
    #a = input('').split(" ")[0]
    noisyImage = np.add(image,noise)
    
    method1 = paperDiscretization.paperDiscretization(noisyImage, coords)
    method1.solve()

    method2 = PDDODiscretization.PDDODiscretization()
    method2.solve()

    # display the array of pixels as an image
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Original Image')
    ax2.imshow(noisyImage, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Noise Image')
    plt.show()



if __name__ == "__main__":
    main()
