import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def main():
    cap = cv2.VideoCapture(0)   # 0 means default webcam
    if not cap.isOpened():
        print("ERROR : Failed to open camera - Quitiing ...") 
        return
    
    hands = mp_hands.Hands(
        model_complexity=0,     # model complexity (0 - lightweight model ,fast but less accurate | 1- full model (default,more accurate, slower) | 3 - Very accurate (but heavy,slower))
        max_num_hands=2,        #  kitne max hands detect karta hai
        min_detection_confidence=0.7,       # 70% sure hone pe hi detect karna
        min_tracking_confidence=0.6         # after detection, 60% sure hone pe hi track karna
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
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
        
        cv2.imshow("Webcam",frame)  # opens up a window and shows ur frame there
        
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            print("Quitting....")
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()