import cv2

# 카메라 영상 받아올 객체 선언 , 설정
capture= cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while True:
    ret, frame =capture.read()
    cv2.imshow("original",frame)
    if cv2.waitKey(1)==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
