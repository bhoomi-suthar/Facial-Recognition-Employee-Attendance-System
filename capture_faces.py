import cv2, os, time

emp_id = input("Enter Employee ID (e.g., E001): ").strip()
emp_name = input("Enter Employee Name (e.g., Alice): ").strip()

save_dir = os.path.join("data", f"{emp_id}_{emp_name}")
os.makedirs(save_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0
TARGET = 30

print("Press 'c' to capture face, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(120,120))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.putText(frame, f"Captured: {count}/{TARGET}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),2)
    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and len(faces) > 0:
        (x,y,w,h) = faces[0]
        face_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (200,200))
        cv2.imwrite(os.path.join(save_dir, f"img_{count:03d}.jpg"), face_resized)
        count += 1
        time.sleep(0.15)
        if count >= TARGET:
            print("Done capturing.")
            break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
