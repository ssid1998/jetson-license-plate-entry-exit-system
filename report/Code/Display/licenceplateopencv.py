	import cv2
	
	frame = cv2.imread("car_emden.jpg")
	cv2.putText(frame, "Willkommen Fahrzeug EMD AB 1234", (40, 50),
	cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cv2.imshow("License Plate Display", frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()