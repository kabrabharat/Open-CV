
import cv2
import numpy as np

cap = cv2.VideoCapture("videoplayback")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

fps_vid = cap.get(cv2.CAP_PROP_FPS)
#print("#####################FPS##################  : ",fps_vid)


# codec for video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#out_name = video_filename[:-4]+"_analysed.mp4"
out_name = "output.mp4"
out = cv2.VideoWriter(out_name, fourcc, fps_vid, (int(width),int(height)))


while cap.isOpened():


	difference = cv2.absdiff(frame1, frame2)
	cv2.imshow("Difference", difference)

	gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

	dilate = cv2.dilate(thresh, None, iterations=3)
	contours, hir = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	#cv2.drawContours(frame1, contours, -1, (0,0,255), 2)
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		if cv2.contourArea(cnt) > 650:
			cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
			

	cv2.imshow("Frame",frame1)
	out.write(frame1)

	frame1 = frame2
	ret, frame2 = cap.read()

	if ret == False:
		break

	if cv2.waitKey(1) == ord('q'):
		break

cv2.destroyAllWindows()
cap.release()

