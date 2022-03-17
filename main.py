import cv2 , dlib
import numpy as np
from imutils import face_utils

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# cudart64_110.dll not found 에러 해결
# 기본값 : 0
# INFO 로그 필터링 : 1
# WARNING 로그 필터링 : 2
# ERROR 로그 필터링 : 3
# 이며 여기서는 '2'로 WARNING을 필터링한 것이다.
import tensorflow as tf


IMG_SIZE = (34,26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('68_landmarks/shape_predictor_68_face_landmarks.dat') # dlib에서 제공한 데이터 셋

model = tf.keras.models.load_model('models/blink_or_not.h5')
model.summary()

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)  
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

# main
cap = cv2.VideoCapture(0) # 캡쳐 객체 생성

while cap.isOpened(): # 캡처 객체 초기화 확인 cap 객체가 지정한 파일이 정상적으로 초기화 -> True
  ret, img_ori = cap.read() # 다음 프레임 읽기 / 프레임 잘 읽었으면 ret = True , img_ori는 프레임 이미지

  if not ret:
    break

  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=1, fy=1)
#   cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None) -> dst
# • src: 입력 영상
# • dsize: 결과 영상 크기. (w, h) 튜플. (0, 0)이면 fx와 fy 값을 이용하여 결정.
# • dst: 출력 영상
# • fx, fy: x와 y방향 스케일 비율(scale factor). (dsize 값이 0일 때 유효)
# • interpolation: 보간법 지정. 기본값은 cv2.INTER_LINEAR

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR2GRAY-> Blue, Green, Red 채널 이미지를 단일 채널, 그레이스케일 이미지로 변경
  # dst = cv2.cvtcolor(src, code, dstCn)는 입력 이미지(src), 색상 변환 코드(code), 출력 채널(dstCn)으로 출력 이미지(dst)을 생성합니다.

  faces = detector(gray) # dectector 을 그레이 스케일 이미지로 변경

  for face in faces: # 얼굴이 여러개 있을 수도 있으니까 (추측)
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)   
    # def shape_to_np(shape, dtype="int"):
	# # initialize the list of (x, y)-coordinates
	# coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# # loop over all facial landmarks and convert them
	# # to a 2-tuple of (x, y)-coordinates
	# for i in range(0, shape.num_parts):
	# 	coords[i] = (shape.part(i).x, shape.part(i).y)

	# # return the list of (x, y)-coordinates
	# return coords
    
    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42]) # number 37~42 왼쪽 눈  (shape_predictor_68_face_landmarks 데이터셋)
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48]) # 43~48 오른쪽 눈

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE) # 모델에 넣기 위해 사이즈 변경
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE) # 모델에 넣기 위해 사이즈 변경
    # eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    cv2.imshow('l', eye_img_l)# 왼쪽 오른쪽 눈의 모습을 화면에 띄워줌
    cv2.imshow('r', eye_img_r)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.# 모델에 넣기 위해 사이즈 변경 , 노멀라이징
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.# 모델에 넣기 위해 사이즈 변경 , 노멀라이징

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r) # 모델에 삽입

    # visualize
    state_l = 'O %.1f' if pred_l > 1 else '- %.1f'
    state_r = 'O %.1f' if pred_r > 1 else '- %.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

  cv2.imshow('result', img) # img 화면에 표시
  if cv2.waitKey(1) == ord('q'): # 1ms 의 지연을 주면서 화면에 표시, q 누르면 break
    break

