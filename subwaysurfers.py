import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pyautogui as pg


interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

noseThreshold = 4 / 10

slideThreshold = 7.5 / 10


rightThreshold = 2.5 / 10
leftThreshold = 7.5 / 10

# x, y = pg.position()

# draw keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    # nose line
    cv2.line(frame, (int(0), int(y * noseThreshold)), (int(x), int(y * noseThreshold)), (0,255,0), 4)

        # slide line
    cv2.line(frame, (int(0), int(y * slideThreshold)), (int(x), int(y * slideThreshold)), (0,255,0), 4)


    # left line
    cv2.line(frame, (int(x * leftThreshold), int(0)), (int(x * leftThreshold), int(y)), (255,0,0), 4)

    # right line
    cv2.line(frame, (int(x * rightThreshold), int(0)), (int(x * rightThreshold), int(y)), (255,0,0), 4)

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)


# 17 keypoints in the order of: 
# [nose, left eye, right eye, left ear, right ear, left shoulder, 
# right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, 
# right knee, left ankle, right ankle]

cap = cv2.VideoCapture(0)

prevNoseFrame = 0
noseIndex = 10  # 0 for nose, 9 or 10 for leftwrist or right wrist respectfully

prevSlideFrame = 0
slideIndex = 10  # 0 for nose, 9 or 10 for leftwrist or right wrist respectfully

prevLeftFrame = 0
leftIndex = 9  # 0 for nose, 9 or 10 for leftwrist or right wrist respectfully

prevRightFrame = 0
rightIndex = 10  # 0 for nose, 9 or 10 for leftwrist or right wrist respectfully

timesFlapped = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.float32)
    
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # goes y, x, z...we care about vertical flap over the line
    # print(keypoints_with_scores[0][0][13][0])

    # jump -> down to up
    if prevSlideFrame < slideThreshold and keypoints_with_scores[0][0][slideIndex][0] > slideThreshold: 
        # timesFlapped += 1
        pg.press('s')  
        # pg.click() #the third argument "1" represents the mouse button
        print("Slide!")

    # slide -> up to down
    if prevNoseFrame > noseThreshold and keypoints_with_scores[0][0][noseIndex][0] < noseThreshold: 
        # timesFlapped += 1
        pg.press('w')  
        # pg.click() #the third argument "1" represents the mouse button
        print("Jump!")

    # left -> cetner to right (MIRRORED!)
    if prevLeftFrame < leftThreshold and keypoints_with_scores[0][0][leftIndex][1] > leftThreshold: 
        # timesFlapped += 1
        pg.press('a')
        # pg.click() #the third argument "1" represents the mouse button
        print("Left!")
    
    # right -> center to right
    if prevRightFrame > rightThreshold and keypoints_with_scores[0][0][rightIndex][1] < rightThreshold: 
        # timesFlapped += 1
        pg.press('d')
        # pg.click() #the third argument "1" represents the mouse button
        print("Right!")

    # Rendering 
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    
    cv2.imshow('MoveNet Lightning', frame)
    prevNoseFrame = keypoints_with_scores[0][0][noseIndex][0]
    prevSlideFrame = keypoints_with_scores[0][0][slideIndex][0]
    prevLeftFrame = keypoints_with_scores[0][0][leftIndex][1]
    prevRightFrame = keypoints_with_scores[0][0][rightIndex][1]

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()