import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR : Failed to open camera - Quitiing ...")
        return
    
    while True:
        ret,frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame,1)
        cv2.imshow("Webcam",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__=="__main__":
    main()
    