import numpy as np
import random
from skimage import color
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def generateRandomPixels():
    """
    Generate random pixel in each 8x8x8 box of the 256x256x256 RGB cube
    """
    #rgbArr = np.zeros((32,32,32,3), dtype=float)
    rgbArr = np.zeros((32,32,32,3))
    for i in range(0,32):
        for j in range(0,32):
            for k in range(0,32):
                #rgbArr[i,j,k,:] = np.random.uniform(i*8,(i+1)*8,3)
                rgbArr[i,j,k,0] = np.random.randint(i*8,(i+1)*8)
                rgbArr[i,j,k,1] = np.random.randint(j*8,(j+1)*8)
                rgbArr[i,j,k,2] = np.random.randint(k*8,(k+1)*8)
    return rgbArr

def generatePairs():
    """ 
    For the each pixel sample, select:
    1 with itself (for 100% closeness case)
    3 close points (up to 2 cubes away)
    3 mid close points (distance is between 3 and 6 cubes)
    3 far points (distance is far than 6 cubes: 7-32)
    """

    global rgbs

    arrlen = 32*32*320
    rgbOriginal = np.zeros((32,32,320,3))
    rgbOpposite = np.zeros((32,32,320,3))

    for i in range(0,32):
        for j in range(0,32):
            for k in range(0,32):
                # First, compare with itself case:

                rgbOriginal[i,j,k*10,:] = rgbs[i,j,k]
                rgbOpposite[i,j,k*10,:] = rgbs[i,j,k]

                # 2, 6 and 31 cubes for close, min and far, correspondingly 
                dist_arr = [0,2,8,31]
                for distIdx in range(1,4):
                    # The range can be on a negative of positive side:
                    #
                    #  _(i-6)________(i-2)___i___(i+2)________(i+6)_
                    #   maxIneg     minIneg      minIpos     maxIpos
                    #
                    minIneg = max(0,i - (dist_arr[distIdx-1]+1))
                    minIpos = min(31,i + (dist_arr[distIdx-1]+1))
                    maxIneg = max(0,i - dist_arr[distIdx-1] - dist_arr[distIdx])
                    maxIpos = min(31,i + dist_arr[distIdx-1] + dist_arr[distIdx])

                    minJneg = max(0,j - (dist_arr[distIdx-1]+1))
                    minJpos = min(31,j + (dist_arr[distIdx-1]+1))
                    maxJneg = max(0,j - dist_arr[distIdx-1] - dist_arr[distIdx])
                    maxJpos = min(31,j + dist_arr[distIdx-1] + dist_arr[distIdx])

                    minKneg = max(0,k - (dist_arr[distIdx-1]+1))
                    minKpos = min(31,k + (dist_arr[distIdx-1]+1))
                    maxKneg = max(0,k - dist_arr[distIdx-1] - dist_arr[distIdx])
                    maxKpos = min(31,k + dist_arr[distIdx-1] + dist_arr[distIdx])

                    # 3 points per close, mid and far
                    for x in range (0,3):
                        if (minIneg > 0) and (minIpos > 0): # get randomly negative or positive part
                            if (random.random() > 0.5):
                                xi = random.randint(maxIneg,minIneg)
                            else:
                                xi = random.randint(minIpos,maxIpos)
                        elif minIneg > 0: # get from negative part
                            xi = random.randint(maxIneg,minIneg)
                        else: # get from positive part
                            xi = random.randint(minIpos,maxIpos)

                        if (minJneg > 0) and (minJpos > 0): # get randomly negative or positive part
                            if (random.random() > 0.5):
                                xj = random.randint(maxJneg,minJneg)
                            else:
                                xj = random.randint(minJpos,maxJpos)
                        elif minJneg > 0: # get from negative part
                            xj = random.randint(maxJneg, minJneg)
                        else: # get from positive part
                            xj = random.randint(minJpos,maxJpos)

                        if (minKneg > 0) and (minKpos > 0): # get randomly negative or positive part
                            if (random.random() > 0.5):
                                xk = random.randint(maxKneg,minKneg)
                            else:
                                xk = random.randint(minKpos,maxKpos)
                        elif minKneg > 0: # get from negative part
                            xk = random.randint(maxKneg,minKneg)
                        else: # get from positive part
                            xk = random.randint(minKpos,maxKpos)

                        rgbOriginal[i,j,k*10 + 1 + (distIdx-1)*3 + x,:] = rgbs[i,j,k]
                        rgbOpposite[i,j,k*10 + 1 + (distIdx-1)*3 + x,:] = rgbs[xi,xj,xk]

    lab1 = color.rgb2lab(rgbOriginal/255.0)
    lab2 = color.rgb2lab(rgbOpposite/255.0)

    sampleData = np.hstack((
        rgbOriginal.reshape(arrlen,3),
        rgbOpposite.reshape(arrlen,3),
        lab1.reshape(arrlen,3),
        lab2.reshape(arrlen,3),
        color.deltaE_ciede2000(lab1,lab2).reshape(arrlen,1)))
    np.savetxt('sampleData.csv', sampleData, delimiter=',',fmt="%s")
    print('Done')
    #plt.scatter(lab1[:,:,:,0], lab1[:,:,:,1], lab1[:,:,:,2], c= 'red')


rgbs = generateRandomPixels()
generatePairs()
#print (rgbs[0:1,0:1,0:1,1])

# https://queirozf.com/entries/matplotlib-pyplot-by-example
# https://matplotlib.org/3.1.1/gallery/mplot3d/scatter3d.html
# https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.figure.Figure.html

#fig = plt.figure()
#ax = fig.add_subplot(32,32,32, projection='3d')
#plt.grid(b=True)
#plt.scatter(lab1[0:5,0:5,0:5,0], rgbs[0:5,0:5,0:5,1], rgbs[0:5,0:5,0:5,2], c= 'red')
#plt.xlim(0,24)
#plt.xticks(np.arange(0, 40, 8))
#plt.yticks(np.arange(0, 40, 8))

#ax.set_title("Random RGB colors")
#ax.set_xlabel('Red')
#ax.set_ylabel('Green')
#ax.set_zlabel('Blue')

#plt.show()