import os
import glob
from skimage import io
import numpy as np
import dlib
import cv2
import sys

if len(sys.argv) < 2 or (sys.argv[1] != 'train' and sys.argv[1] != 'test'):
    raise ValueError('usage: python data/face_landmark_detection.py [train|test]')

phase = sys.argv[1]
dataset_path = 'datasets/face/'
faces_folder_path = os.path.join(dataset_path, phase + '_img/')
predictor_path = os.path.join(dataset_path, 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def draw_landmarks(im, landmarks):
    CIRCLE_SIZE = 1
    FONT_SCALE = 1
    THICKNESS_S = 2
    im = im.copy()
    #  0 - 16: head
    for idx, point in enumerate(landmarks[0:17]):
        pos = (point[0], point[1])
        #cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=FONT_SCALE, color=(0, 0, 255))
        cv2.circle(im, pos, CIRCLE_SIZE, color=(255, 0, 0), thickness=THICKNESS_S)
    # 17 - 21: left eye brow
    # 22 - 26: right eye brow
    for idx, point in enumerate(landmarks[17:27]):
        pos = (point[0], point[1])
        #cv2.putText(im, str(idx), pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=FONT_SCALE,color=(0, 0, 255))
        cv2.circle(im, pos, CIRCLE_SIZE, color=(0, 255, 0), thickness=THICKNESS_S)
    # 27 - 35: nose
    for idx, point in enumerate(landmarks[27:36]):
        pos = (point[0], point[1])
        #cv2.putText(im, str(idx), pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=FONT_SCALE,color=(0, 0, 255))
        cv2.circle(im, pos, CIRCLE_SIZE, color=(0, 0, 255), thickness=THICKNESS_S)
    # 36 - 41: left eye
    # 42 - 47: right eye
    for idx, point in enumerate(landmarks[36:48]):
        pos = (point[0], point[1])
        #cv2.putText(im, str(idx), pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=FONT_SCALE,color=(0, 0, 255))
        cv2.circle(im, pos, CIRCLE_SIZE, color=(0, 255, 255), thickness=THICKNESS_S)
    # 48 - 68: lips
    for idx, point in enumerate(landmarks[48:68]):
        pos = (point[0], point[1])
        #cv2.putText(im, str(idx), pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=FONT_SCALE,color=(0, 0, 255))
        cv2.circle(im, pos, CIRCLE_SIZE, color=(255, 0, 255), thickness=THICKNESS_S)
    return im

img_paths = sorted(glob.glob(faces_folder_path + '*'))
for i in range(len(img_paths)):
    f = img_paths[i]
    print("Processing video: {}".format(f))
    save_path = os.path.join(dataset_path, phase + '_keypoints', os.path.basename(f))
    debug_path = os.path.join(dataset_path, phase + '_debug', os.path.basename(f))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(debug_path):
        os.makedirs(debug_path)

    for img_name in sorted(glob.glob(os.path.join(f, '*.jpg'))):
        img = io.imread(img_name)
        dets = detector(img, 1)
        if len(dets) > 0:
            shape = predictor(img, dets[0])
            points = np.empty([68, 2], dtype=int)
            for b in range(68):
                points[b,0] = shape.part(b).x
                points[b,1] = shape.part(b).y

            save_name = os.path.join(save_path, os.path.basename(img_name)[:-4] + '.txt')
            debug_name = os.path.join(debug_path, os.path.basename(img_name)[:-4] + '.jpg')
            np.savetxt(save_name, points, fmt='%d', delimiter=',')
            img_with_landmarks = draw_landmarks(img, points)
            io.imsave(debug_name, img_with_landmarks)
