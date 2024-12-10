import numpy as np
from skimage import color
from JMLanfis import calculateAnfisOutput

def smooth_step(x):
	a = 0.7
	b = 1.00
	t = min(max((x-a)/(b-a),0),1.0)
	y = t*t*(3-2*t)
	return y

rgbA = [204,0,102]
rgbB = [102,255,178]

rgbArr = np.uint8([[rgbA]])
lab1 = color.rgb2lab(rgbArr/255.0)

rgbArr = np.uint8([[rgbB]])
lab2 = color.rgb2lab(rgbArr/255.0)

l1 = lab1[0][0][0]/100.0
a1 = (lab1[0][0][1]+110.0)/210.0
b1 = (lab1[0][0][2]+110.0)/210.0
l2 = lab2[0][0][0]/100.0
a2 = (lab2[0][0][1]+110.0)/210.0
b2 = (lab2[0][0][2]+110.0)/210.0

output = calculateAnfisOutput([l1,a1,b1,l2-l1,a2-a1,b2-b1])
closeness = smooth_step(output)
print("Output: " +str(output) + " Smooth Step: " +str(closeness))
