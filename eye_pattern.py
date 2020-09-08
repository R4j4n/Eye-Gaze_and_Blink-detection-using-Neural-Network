import cv2, dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
import winsound

IMG_SIZE = (64,56)
B_SIZE = (34, 26)
margin = 95
class_labels = ['center','left', 'right'] 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

font_letter = cv2.FONT_HERSHEY_PLAIN
model = load_model('models/gazev3.1.h5')
model_b = load_model('models/blinkdetection.h5')


def detect_gaze(eye_img):
    pred_l = model.predict(eye_img)
    accuracy = int(np.array(pred_l).max() * 100)
    gaze = class_labels[np.argmax(pred_l)]
    return gaze


def detect_blink(eye_img):
    pred_B = model_b.predict(eye_img)
    status = pred_B[0][0]
    status = status*100
    status = round(status,3)
    return  status

   
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
cap = cv2.VideoCapture(0)
# pattern = []
# frames = 10
# pattern_length = 0
frames_to_blink = 6
blinking_frames = 0
while cap.isOpened():
    output = np.zeros((900,820,3), dtype="uint8")
    ret, img = cap.read()
    img = cv2.flip(img,flipCode = 1)
    h,w = (112,128)	
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    faces = detector(gray)

    for face in faces:
        shapes = predictor(gray, face)
        

        for n in range(36,42):
            x= shapes.part(n).x
            y = shapes.part(n).y
            next_point = n+1
            if n==41:
                next_point = 36 
            
            x2 = shapes.part(next_point).x
            y2 = shapes.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(0,69,255),2)

        for n in range(42,48):
            x= shapes.part(n).x
            y = shapes.part(n).y
            next_point = n+1
            if n==47:
                next_point = 42 
            
            x2 = shapes.part(next_point).x
            y2 = shapes.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(153,0,153),2)
        shapes = face_utils.shape_to_np(shapes)
        #~~~~~~~~~~~~~~~~~56,64 EYE IMAGE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])
        #~~~~~~~~~~~~~~~~FOR THE EYE FINAL_WINDOW~~~~~~~~~~~~~~~~~~~~~~#
        eye_img_l_view = cv2.resize(eye_img_l, dsize=(128,112))
        eye_img_l_view = cv2.cvtColor(eye_img_l_view,cv2.COLOR_BGR2RGB)
        eye_img_r_view = cv2.resize(eye_img_r, dsize=(128,112))
        eye_img_r_view = cv2.cvtColor(eye_img_r_view, cv2.COLOR_BGR2RGB)
        #~~~~~~~~~~~~~~~~~FOR THE BLINK DETECTION~~~~~~~~~~~~~~~~~~~~~~~
        eye_blink_left = cv2.resize(eye_img_l.copy(), B_SIZE)
        eye_blink_right = cv2.resize(eye_img_r.copy(), B_SIZE)
        eye_blink_left_i = eye_blink_left.reshape((1, B_SIZE[1], B_SIZE[0], 1)).astype(np.float32) / 255.
        eye_blink_right_i = eye_blink_right.reshape((1, B_SIZE[1], B_SIZE[0], 1)).astype(np.float32) / 255.
        #~~~~~~~~~~~~~~~~FOR THE GAZE DETECTIOM~~~~~~~~~~~~~~~~~~~~~~~~#
        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_input_g = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        #~~~~~~~~~~~~~~~~~~PREDICTION PROCESS~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        status_l = detect_blink(eye_blink_left_i)
        gaze =  detect_gaze(eye_input_g)
        if gaze == class_labels[1]:
            blinking_frames += 1
            if blinking_frames == frames_to_blink:
                # pattern_length +=1
                winsound.Beep(1000,250)
                # pattern.append(1)
        elif gaze == class_labels[2]:
            blinking_frames += 1
            if blinking_frames == frames_to_blink:
                # pattern_length +=1
                winsound.Beep(1000,250)
                # pattern.append(2)

        elif status_l < 0.1:
            blinking_frames += 1
            if blinking_frames == frames_to_blink:
                # pattern_length +=1
                # pattern.append(3)
                winsound.Beep(1000,250)
        else:
            blinking_frames = 0
        #~~~~~~~~~~~~~~~~~~~~~~~FINAL_WINDOWS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        output = cv2.line(output,(400,200), (400,0),(0,255,0),thickness=2)
        cv2.putText(output,"LEFT EYE GAZE",(10,180), font_letter,1, (255,255,51),1)
        cv2.putText(output,"LEFT EYE OPENING %",(200,180), font_letter,1, (255,255,51),1)
        cv2.putText(output,"RIGHT EYE GAZE",(440,180), font_letter,1, (255,255,51),1)
        cv2.putText(output,"RIGHT EYE OPENING %",(621,180), font_letter,1, (255,255,51),1)		
        if status_l < 10 :
            cv2.putText(output,"---BLINKING----",(250,300), font_letter,2, (153,153,255),2)
        output[0:112, 0:128] = eye_img_l_view
        cv2.putText(output, gaze,(30,150), font_letter,2, (0,255,0),2)
        output[0:112, margin+w:(margin+w)+w] = eye_img_l_view
        cv2.putText(output,(str(status_l)+"%"),((margin+w),150), font_letter,2, (0,0,255),2)
        output[0:112, 2*margin+2*w:(2*margin+2*w)+w] = eye_img_r_view
        cv2.putText(output, gaze,((2*margin+2*w)+30,150), font_letter,2, (0,0,255),2)
        output[0:112, 3*margin+3*w:(3*margin+3*w)+w] = eye_img_r_view
        cv2.putText(output, (str(status_l)+"%"),((3*margin+3*w),150), font_letter,2, (0,0,255),2)
        output[235+100:715+100, 80:720] = img
        
        cv2.imshow('result',output)
    if cv2.waitKey(1) == ord('q') : 
        break
cap.release()
cv2.destroyAllWindows()    
# print(pattern)
