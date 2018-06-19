import numpy as np
import cv2
import time
import RPi.GPIO as GPIO
    #import dcmotorstest2 as motor
        #Difference Variable
minDiff = 10000
minSquareArea = 5000
match = -1
#hello = 1
        #Frame width & Height
w=640
h=480

        #Reference Images Display name & Original Name
ReferenceImages = ["Start.bmp","Wait.bmp","Stop2.bmp"]
ReferenceTitles = ["Start","Wait5sec","Stop"]
cap=cv2.VideoCapture(-1)
    #cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 5)
    #cap = cv2.VideoCapture(1)
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings (False)
GPIO.setup(7, GPIO.OUT)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)
pwma=GPIO.PWM(07,100)
pwmd=GPIO.PWM(12,100)
    #pwma.start(35)
    #pwmd.start(35)
    #GPIO.output(7, GPIO.LOW)
    #GPIO.output(12, GPIO.LOW)
    #time.sleep(3)

        #define class for References Images
class Symbol:
        def __init__(self):
                self.img = 0
                self.name = 0
                
symbol= [Symbol() for i in range(3)]
        #define class instances (6 objects for 6 different images)



def readRefImages():
            for count in range(3):
                image = cv2.imread(ReferenceImages[count], cv2.COLOR_BGR2GRAY)
                symbol[count].img = cv2.resize(image,(w/2,h/2),interpolation = cv2.INTER_AREA)
                symbol[count].name = ReferenceTitles[count]
                #cv2.imshow(symbol[count].name,symbol[count].img);


def order_points(pts):
                # initialzie a list of coordinates that will be ordered
                # such that the first entry in the list is the top-left,
                # the second entry is the top-right, the third is the
                # bottom-right, and the fourth is the bottom-left
                rect = np.zeros((4, 2), dtype = "float32")

                # the top-left point will have the smallest sum, whereas
                # the bottom-right point will have the largest sum
                s = pts.sum(axis = 1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]

                # now, compute the difference between the points, the
                # top-right point will have the smallest difference,
                # whereas the bottom-left will have the largest difference
                diff = np.diff(pts, axis = 1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                # return the ordered coordinates
                return rect

def four_point_transform(image, pts):
                # obtain a consistent order of the points and unpack them
                # individually
                rect = order_points(pts)
                (tl, tr, br, bl) = rect

                maxWidth = w/2
                maxHeight = h/2

                dst = np.array([
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype = "float32")

                # compute the perspective transform matrix and then apply it
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

                # return the warped image
                return warped


def auto_canny(image, sigma=0.33):
                # compute the median of the single channel pixel intensities
                v = np.median(image)

                # apply automatic Canny edge detection using the computed median
                lower = int(max(0, (1.0 - sigma) * v))
                upper = int(min(255, (1.0 + sigma) * v))
                edged = cv2.Canny(image, lower, upper)

                # return the edged image
                return edged


def resize_and_threshold_warped(image):
                #Resize the corrected image to proper size & convert it to grayscale
                #warped_new =  cv2.resize(image,(w/2, h/2))
                warped_new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                #Smoothing Out Image
                blur = cv2.GaussianBlur(warped_new_gray,(5,5),0)

                #Calculate the maximum pixel and minimum pixel value & compute threshold
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
                threshold = (min_val + max_val)/2

                #Threshold the image
                ret, warped_processed = cv2.threshold(warped_new_gray, threshold, 255, cv2.THRESH_BINARY)

                #return the thresholded image
                return warped_processed


def signs(flag,shm):
	    #kala=1
            #Font Type
            font = cv2.FONT_HERSHEY_SIMPLEX

            # initialize the camera and grab a reference to the raw camera capture
            #video = cv2.VideoCapture(0)
            #video.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            #Windows to display frames
            #cv2.namedWindow("Main Frame", cv2.WINDOW_AUTOSIZE)
            #cv2.namedWindow("Matching Operation", cv2.WINDOW_AUTOSIZE)
            #cv2.namedWindow("Corrected Perspective", cv2.WINDOW_AUTOSIZE)
            #cv2.namedWindow("Contours", cv2.WINDOW_AUTOSIZE)

            #Read all the reference images
            readRefImages()

            # capture frames from the camera
            while True:
                    ret, frames = cap.read()
                    grays = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                    blurreds = cv2.GaussianBlur(grays,(3,3),0)

                    #Detecting Edges
                    edgess = auto_canny(blurreds)

                    #Contour Detection & checking for squares based on the square area
                    contourss, hierarchy = cv2.findContours(edgess,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                    for cnt12 in contourss:
                            approxs = cv2.approxPolyDP(cnt12,0.01*cv2.arcLength(cnt12,True),True)
					

			    if len(approxs)!=4 and flag!=0:
				    main(shm)
                            elif len(approxs)==4:
                                    areas = cv2.contourArea(approxs)

                                    if areas > minSquareArea:
                                            #cv2.drawContours(OriginalFrame,[approx],0,(0,0,255),2)
                                            warpeds = four_point_transform(frames, approxs.reshape(4, 2))
                                            warped_eqs = resize_and_threshold_warped(warpeds)
                                            #print "PARAKATW TO WARPED_EQ"
                                            #print warped_eq
                                            #print len(warped_eq)
                                            #print warped_eq.dtype
                                            #print warped_eq.size
                                            #print warped_eq[100,100]
                                            #print "PARAKATW TO SYMBOL[i]"
                                            #print symbol[0].img
                                            #print len(symbol[0].img)
                                            #print symbol[0].img.dtype
                                            #print symbol[0].img.size
                                            #print symbol[0].img[100,100]
                                            #print "TELOS"


                                            for i in range(3):
                                                diffImgs = cv2.bitwise_xor(warped_eqs, symbol[i].img)
                                                diffs = cv2.countNonZero(diffImgs);
                                                #print diff
                                                #print symbol[0].name
                                                #print symbol[1].name
                                                #print symbol[2].name
                                                #print symbol[i].name, diff

                                                if diffs < minDiff:
                                                    matchs = i
                                                    #print "i="
                                                    #print i
                                                    print diffs

                                                    #print symbol[i].name, diff
                                                    #print approx.reshape(4,2)[0]
                                                    #cv2.putText(OriginalFrame,symbol[i].name, (10,30), font, 1, (255,0,255), 2, cv2.LINE_AA)
                                                    if diffs < 4500 and diffs > 3000 and flag!=0 and shm==0:
                                                        print "WAIT"
														pwma.start(0)
														pwmd.start(0)
														GPIO.output(16, GPIO.LOW)
														GPIO.output(18, GPIO.LOW)
														GPIO.output(7, GPIO.LOW)
														GPIO.output(13, GPIO.LOW)
														GPIO.output(15, GPIO.LOW)
														GPIO.output(12, GPIO.LOW)
														time.sleep(5)
														pwma.start(30)
														pwmd.start(32.8)
														GPIO.output(16, GPIO.HIGH)
														GPIO.output(18, GPIO.LOW)
														GPIO.output(7, GPIO.HIGH)
														GPIO.output(13, GPIO.HIGH)
														GPIO.output(15, GPIO.LOW)
														GPIO.output(12, GPIO.HIGH)
														time.sleep(0.5)
														pwma.start(0)
														pwmd.start(0)
														time.sleep(1.5)
														shm = 1
														main(shm)
													elif diffs < 6200 and diffs > 4500 and flag==0:
														print "START"
														pwma.start(0)
														pwmd.start(0)
														time.sleep(3)
														#pwma.start(30)
														#pwmd.start(32.8)
														#GPIO.output(16, GPIO.HIGH)
														#GPIO.output(18, GPIO.LOW)
														#GPIO.output(7, GPIO.HIGH)
														#GPIO.output(13, GPIO.HIGH)
														#GPIO.output(15, GPIO.LOW)
														#GPIO.output(12, GPIO.HIGH)
														#time.sleep(0.5)
														pwma.start(0)
														pwmd.start(0)
														time.sleep(0.5)
														main(shm)
													elif diffs > 8000 and diffs < 9999 and flag!=0:
														print "STOP"
														GPIO.output(16, GPIO.LOW)
														GPIO.output(18, GPIO.LOW)
														GPIO.output(7, GPIO.LOW)
														GPIO.output(13, GPIO.LOW)
														GPIO.output(15, GPIO.LOW)
														GPIO.output(12, GPIO.LOW)
														cap.release()
														cv2.destroyAllWindows()
														GPIO.cleanup()
                                                    #cv2.putText(OriginalFrame,symbol[i].name, tuple(approx.reshape(4,2)[0]), font, 1, (255,0,255), 2, cv2.LINE_AA)
                                                    diffs = minDiff
                                                    #print kala
                                                    #kala=kala+1
                                                    break;
#				else:
#					main()


def main(shm):
	#cap = cv2.VideoCapture(-1)
        #kala=1
        #Font Type
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #readRefImages()
        while(True):	
            #f=1
            # Capture frame-by-frame
            #time.sleep(2)
            ret, frame = cap.read()
            #while(frame is None):
            #	time.sleep(1)
            #	ret, frame = cap.read()
            #	time.sleep(1)
            #	print f
            #	f=f+1
            #print ret
            #print frame
            #print("wait")
            #time.sleep(10)
            # Our operations on the frame come here
            crop = frame[180:320, 0:638]
            crop2=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            th,crop2 = cv2.threshold(crop2,0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            previous = cv2.GaussianBlur(crop2, (5,5),0)
	    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    #blurred = cv2.GaussianBlur(gray,(3,3),0)
            #edges = auto_canny(blurred)
            #edges = cv2.Canny(previous,100,200)
            contours, hierarchy = cv2.findContours(crop2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	    #contours1, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
            i=0
            #for x in range(0,638):
                        #cv2.circle(previous,(x,0),5,(0,0,255),-1)
                        #cv2.circle(previous,(x,140),5,(0,0,255),-1)
            #GPIO.output(7, GPIO.LOW)
            #GPIO.output(12, GPIO.LOW)
            #time.sleep(2)
            for cnt in contours:
				moments = cv2.moments(cnt)
                if moments['m00']!=0:
                    cx = int(moments['m10']/moments['m00'])         # cx = M10/M00
                    cy = int(moments['m01']/moments['m00'])         # cy = M01/M00
                    moment_area = moments['m00']                    # Contour area from moment
                    contour_area = cv2.contourArea(cnt)             # Contour area using in_built function
                    perimeter = cv2.arcLength(cnt,True)
                    #epsilon = 0.1*cv2.arcLength(cnt,True)
                    #approx = cv2.approxPolyDP(cnt,perimeter,True)
                    hull = cv2.convexHull(cnt)
                    k = cv2.isContourConvex(cnt)
                    x,y,w,h = cv2.boundingRect(cnt)
                    rows,cols = previous.shape[:2]
                    #[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
                    #lefty = int((-x*vy/vx) + y)
                    #righty = int(((cols-x)*vy/vx)+y)
                    #cv2.line(previous,(0,638),(0,140),(0,255,0),2)
                    #cv2.rectangle(perimeter,(x,y),(x+w,y+h),(0,255,0),2) 
                    #cv2.drawContours(previous, contours, -1, (0,255,0), 3)   # draw contours in green color
                    #cnt = contours[4]
                    cv2.drawContours(previous, [cnt], 0, (0,255,0), 3)
                    #cv2.circle(perimeter,(cx,cy),5,(0,0,255),-1)      # draw centroids in red color
                    #cv2.circle(previous,(637,139),5,(0,0,255),-1)
                    #cv2.circle(edges,(cx,cy),5,(255,0,0),-1)
                    #lines = cv2.HoughLines(previous,1,np.pi/180,200) 
                    px = previous[cy,cx]
                    if px == 255 :
                        i=i+1
                        cv2.circle(previous,(cx,cy),5,(0,0,255),-1)
                        #cv2.circle(edges,(cx,cy),5,(255,0,0),-1)
                    #if previous[70,319]==255:
                        #print 'eutheia'

            #if cy > 60 and cy < 90:
            if cx < 212:
                    print 'LEFT LEFT LEFT LEFT'
                    if cx < 100:
                            l=100
                    elif cx < 200 and cx >= 100:
                            l=75
                    elif cx >=200:
                            l=30
                    pwma.start(l)
                    pwmd.start(l)	
                    GPIO.output(13, GPIO.HIGH)
                    GPIO.output(15, GPIO.LOW)
                    GPIO.output(12, GPIO.HIGH)
                    GPIO.output(7, GPIO.HIGH)
                    GPIO.output(16, GPIO.LOW)
                    GPIO.output(18, GPIO.HIGH)
                    time.sleep(.010)
                    #motor.stop()
                    #time.sleep(.000001)
                    #time.sleep(.000000000001)
            elif cx >=212 and cx <=426 :
                    print 'EUTHEIA EUTHEIA EUTHEIA EUTHEIA'
                    pwma.start(30)
                    pwmd.start(32.8)
                    GPIO.output(16, GPIO.HIGH)
                    GPIO.output(18, GPIO.LOW)
                    GPIO.output(7, GPIO.HIGH)
                    GPIO.output(13, GPIO.HIGH)
                    GPIO.output(15, GPIO.LOW)
                    GPIO.output(12, GPIO.HIGH)
                    time.sleep(.01)
                    #motor.stop()
                    #time.sleep(.000001)
                    #time.sleep(.00000000000000000000000000000000000000000000000000000000000000000000000001)
            elif cx>426:
                    print 'RIGHT RIGHT RIGHT RIGHT'
                    if cx > 550 :
                            p = 100
                    elif cx > 400 :
                            p = 75
                    else:
                            p = 30
                    pwma.start(p)
                    pwmd.start(p)
                    GPIO.output(16, GPIO.HIGH)
                    GPIO.output(18 ,GPIO.LOW)
                    GPIO.output(7, GPIO.HIGH)
                    GPIO.output(12, GPIO.HIGH)
                    GPIO.output(13, GPIO.LOW)
                    GPIO.output(15, GPIO.HIGH)
                    time.sleep(.010)
                    #motor.stop()
                    #time.sleep(.000001)    
                    #time.sleep(.000000000001)
            if i!=1:
					pwma.start(30)
					pwmd.start(32.8)
                    GPIO.output(7, GPIO.HIGH)
                    GPIO.output(12, GPIO.HIGH)
                    GPIO.output(13, GPIO.HIGH)
                    GPIO.output(15, GPIO.LOW)
                    GPIO.output(16, GPIO.HIGH)
                    GPIO.output(18, GPIO.LOW)
					time.sleep(0.0000000000000000000000000000000001)
                    print("TELOS TELOS TELOS")
					pwma.start(0)
					pwmd.start(0)
					time.sleep(0.1)
					flag=1
					signs(flag,shm)
                    #time.sleep(30)
                    #pwmd.start(0)

            print 'to synolo twn shmeiwn einai:%d'%i
            print 'to cx einai:%d'%cx
            print 'to cy einai:%d'%cy
            print 'to xrwma einai:%d' %px
            #print 'grapse kati'
            #print moments
            #cv2.imshow('Edges',edges)
            #cv2.imshow("Previous",previous)
            GPIO.output(7, GPIO.LOW)
            GPIO.output(12, GPIO.LOW)
            GPIO.output(13, GPIO.LOW)
            GPIO.output(15, GPIO.LOW)
            GPIO.output(16, GPIO.LOW)
            GPIO.output(18, GPIO.LOW)
            time.sleep(.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001)
	    #signs()	
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

flag = 0
shm = 0
#hello = 1
signs(flag,shm)
#main()
#cap.release()
#cv2.destroyAllWindows()
#GPIO.cleanup()


