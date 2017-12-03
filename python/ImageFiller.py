import cv2
import numpy as np
import sys
from JMLanfis import calculateAnfisOutput

# get current threshold
print('recursion limit ',sys.getrecursionlimit(),' is increased to 10,000')
# there's need to increase the recursion
sys.setrecursionlimit(10000)

windowNo = 1
windowSize = 500
# pictureArea is used to mark areas where we've been before
pictureArea = np.zeros((windowSize,windowSize))
iterCnt = 0

# mouse callback function
def mouse_callback(event,x,y, flags, param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		global iterCnt
		iterCnt = 0
		print('Clicked')
		rgbP = img[y,x]
		print(rgbP)
		rgbArr = np.uint8([[rgbP]])
		hsvP = cv2.cvtColor(rgbArr,cv2.COLOR_BGR2HSV)
		print(hsvP)
		print('H=',hsvP[0][0][0],'(',hsvP[0][0][0]/180,')')
		print('S=',hsvP[0][0][1],'(',hsvP[0][0][1]/255,')')
		print('V=',hsvP[0][0][2],'(',hsvP[0][0][2]/255,')')
		cv2.circle(img,(x,y),2,(0,0,255),-1)
		lookForNeighbours(y,x,0.6)

		
# check if this coordinate is related to the given color
def lookForNeighbours(x,y,thres):
	global iterCnt
	iterCnt += 1
	if (iterCnt >5000):
		return
	# process only pixels inside the area
	if ((x<1) | (y<1) | (x>windowSize) | (y>windowSize)):
		return
	elif (pictureArea[x,y] != 0):
		return
	else:	
		#get color in this position
		rgbP = img[x,y]
		rgbArr = np.uint8([[rgbP]])
		hsvP = cv2.cvtColor(rgbArr,cv2.COLOR_BGR2HSV)
		# calculate color closeness with ANFIS
		# OpenCV provides HSV in [0-180,0-255,0-255] range
		h = hsvP[0][0][0]*1.0/180
		s = hsvP[0][0][1]*1.0/255
		v = hsvP[0][0][2]*1.0/255
		closeness = calculateAnfisOutput([h,s,v])
		print('hsv(',hsvP[0][0][0],',',hsvP[0][0][1],',',hsvP[0][0][2],') -> ',closeness)
		
		if (closeness > thres):
			# mark this coordinate
			pictureArea[x,y] = 1
			# mark this pixel as red
			img[x,y] = [0,255,0]
			# check all 8 directions.
			lookForNeighbours(x-1,y-1,thres)
			lookForNeighbours(x-1,y-1,thres)
			lookForNeighbours(x-1,y,thres)
			lookForNeighbours(x-1,y+1,thres)
			lookForNeighbours(x+1,y-1,thres)
			lookForNeighbours(x+1,y,thres)
			lookForNeighbours(x+1,y+1,thres)
			lookForNeighbours(x,y-1,thres)
			lookForNeighbours(x,y+1,thres)
		
		
#img = cv2.imread('dave.jpg')
img = cv2.imread('D:\\Aeroimages\\rgb1.jpg')
startPos = windowSize*(windowNo-1)+1
endPos   = startPos + windowSize
img = img[startPos:endPos,startPos:endPos]

cv2.namedWindow('image')
cv2.setMouseCallback('image',mouse_callback)
while(1):
	cv2.imshow("image",img)
	if cv2.waitKey(20) & 0xFF == 27:
		break
cv2.destroyAllWindows

'''
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel = np.ones((4,4),np.uint8)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

edges = cv2.Canny(opening, threshold1=100, threshold2=200, apertureSize=3)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

for i in range(0,lines.shape[0]):
	print('for loop ',lines[i,0])
	x1,y1,x2,y2 = lines[i,0]
	cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)

	
cv2.imwrite('houghlines5.jpg',edges)
cv2.imshow('Original',img)
cv2.imshow('Result',opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''