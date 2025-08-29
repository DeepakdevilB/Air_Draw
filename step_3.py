# STEP 3 - DRAW A SEPARATE CANVAS ONLY WHEN INDEX FINGER IS UP
import cv2
import mediapipe as mp
import numpy as np

DRAW_THICKNESS = 8      # line ki motai
EMA_ALPHA = 0.6         # for smoothening and stability

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def to_pixels(hand_landmarks,w,h):
    return [(int(lm.x*w),int(lm.y*h)) for lm in hand_landmarks.landmark]

def fingers_up(pts):
    index_up = pts[8][1] < pts[6][1]
    middle_up = pts[12][1] < pts[10][1]
    return index_up,middle_up

def overlay(frame,canvas):
    gray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _,inv = cv2.threshold(gray,20,255,cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv,cv2.COLOR_GRAY2BGR)
    base = cv2.bitwise_and(frame,inv)
    return cv2.bitwise_or(base,canvas)

def main():
    cap = cv2.VideoCapture(0)   # 0 means default webcam
    
    if not cap.isOpened():
        print("ERROR : Failed to open camera - Quitiing ...")
        return
    
    cap.set(3,1920) #width of frame captured
    cap.set(4,1080)         # height
    
    # creating height and width of canvas
    ret, frame = cap.read()
    if not ret:
        print("Camera Error")
        return
    h,w = frame.shape[:2]
    canvas = np.zeros((h,w,3),dtype=np.uint8)
    color = (0,0,255)       #red BGR
    
    smx = smy = None        # smx - start mouse X / smy - start mouse Y
    prev = None
    
    hands = mp_hands.Hands(
        model_complexity=1,     # model complexity (0 - lightweight model ,fast but less accurate | 1- full model (default,more accurate, slower) | 3 - Very accurate (but heavy,slower))
        max_num_hands=1,        #  kitne max hands detect karta hai
        min_detection_confidence=0.7,       # 70% sure hone pe hi detect karna
        min_tracking_confidence=0.6         # after detection, 60% sure hone pe hi track karna
    )

    while True:
        is_true, frame = cap.read()
        if not is_true:
            break
        fh,fw = frame.shape[:2]
        
        frame = cv2.flip(frame,1)   # src,flipcode (0 for vertical, 1 for horizontal , 2 for both)
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)     # cv2 gives BGR(blue,green,red) by default but mediapipe needs RGB to process the frame properly
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),       # for custom styling of landmark points and lines, control humare saath mai aagya
                    mp_styles.get_default_hand_connections_style(),
                )
            
            pts = to_pixels(results.multi_hand_landmarks[0],fw,fh)
            idx_tip = pts[8]
            index_up,middle_up = fingers_up(pts)
            
            if index_up and not middle_up:
                x,y = idx_tip
                x = max(0,min(x,fw-1))
                y = max(0,min(y,fh-1))
                
                if smx is None:
                    smx,smy = x,y
                else:
                    smx = int(EMA_ALPHA * smx + (1-EMA_ALPHA)*x)
                    smy = int(EMA_ALPHA * smy + (1-EMA_ALPHA)*y)
                    
                if prev is None:
                    prev = (smx,smy)
                
                cv2.line(canvas,prev,(smx,smy),color,DRAW_THICKNESS)
                prev = (smx,smy)
            else:
                prev = None
        
        out = overlay(frame,canvas)
        cv2.putText(out,"Draw: index up | Clear : c | Quit : q",
                    (10,30),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
        
                    
        cv2.imshow("Webcam",out)  # opens up a window and shows ur frame there
        
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            print("Quitting....")
            break
        if key == ord('c'):
            print("Clearing Canvas...")
            canvas[:]=0

    cap.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()