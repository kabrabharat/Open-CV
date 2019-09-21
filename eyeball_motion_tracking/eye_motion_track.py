import cv2
import numpy as np

cap = cv2.VideoCapture(0)
haar_cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



while True:

	ret, frame = cap.read()

	if ret == False:
		break
	image_gray = cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2GRAY)
    
	faces_rects = haar_cascade_face.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 5)
	
	img = []
	th = []
	for (x,y,w,h) in faces_rects:
		
		#image_crop.append(frame[y:y+h, x:x+h])
		img_crop1 = image_gray[y:y+h, x:x+int(w/2)]
		#img_crop2 = image_gray[y:y+h, int(x+w/2):x+w]
		#cv2.imshow("Frame_eye", img_crop)
		# blur
		
		eye_rect_left = haar_cascade_eye.detectMultiScale(img_crop1, scaleFactor = 1.2, minNeighbors = 5)	
		#eye_rect_right = haar_cascade_eye.detectMultiScale(img_crop2, scaleFactor = 1.2, minNeighbors = 5)
		#blur = cv2.GaussianBlur(img_crop, (7,7), 0)
		#img.append(img_crop)
		for (x,y,w,h) in eye_rect_left:
			eye = img_crop1[y+10:y+h-10, x:x+w]
			
			blur = cv2.GaussianBlur(eye, (7,7), 0)

			_, thresh = cv2.threshold(blur, 45, 250, cv2.THRESH_BINARY_INV)
			
			contours, hir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			#kernel = np.ones((5,5), np.uint8)
			#img_erosion = cv2.erode(thresh, kernel, iterations=2) 
			#img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)
			#img_dilation2 = cv2.dilate(thresh, kernel, iterations=2)

			for cnt in contours:
				cv2.drawContours(eye, [cnt], -1, (0,255,255), 3)

			cv2.rectangle(img_crop1, (x, y), (x+w, y+h), (0, 255, 255), 2)
			cv2.imshow("EYE", eye)
			#cv2.imshow("Dil without err", img_dilation2)
			#cv2.imshow("Dil", img_dilation)
			#cv2.imshow("Erosion", img_erosion)
			cv2.imshow("Eye_thresh",thresh)
		#for (x,y,w,h) in eye_rect_left:
		#	cv2.rectangle(img_crop2, (x, y), (x+w, y+h), (0, 255, 255), 2)
			print(np.sum(thresh))
			#if np.sum(img_dilation2)>400000:
			#	print(img_dilation2)
			#	print("\nBlinked") 
		
		cv2.imshow("Frame_left", img_crop1)
		#cv2.imshow("Frame_right", img_crop2)
		#_, thresh = cv2.threshold(img_crop, 5, 255, cv2.THRESH_BINARY_INV)
		#cv2.imshow("Eye_thresh",thresh)
		#th.append(thresh)
		
		#cv2.rectangle(frame, (eye_rect_left[0][0], eye_rect_left[0][1]), (eye_rect_left[0][0]+eye_rect_left[0][2], eye_rect_left[0][1]+eye_rect_left[0][3]), (0, 255, 0), 2)
		#cv2.rectangle(frame, (eye_rect_right[0][0], eye_rect_right[0][1]), (eye_rect_right[0][0]+eye_rect_right[0][2], eye_rect_right[0][1]+eye_rect_right[0][3]), (0, 255, 0), 2)
		
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#	if len(img)==2:
#		cv2.imshow("Eye_thresh1",th[0])
#		cv2.imshow("Frame_eye1", img[0])
		#cv2.imshow("Eye_thresh2",th[1])
		#cv2.imshow("Frame_eye2", img[1])
	cv2.imshow("Frame", frame)	
	
cap.release()
cv2.destroyAllWindows()
