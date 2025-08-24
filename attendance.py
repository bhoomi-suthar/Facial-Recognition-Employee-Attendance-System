import os, csv, json, datetime, cv2

MODEL_PATH = "model/trainer.yml"
LABELS_PATH = "model/labels.json"
CSV_PATH = "attendance.csv"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

with open(LABELS_PATH,"r") as f:
    label_map = {int(k):v for k,v in json.load(f).items()}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date","time","employee_id","employee_name"])

today = datetime.date.today().isoformat()
marked_today = set()
with open(CSV_PATH,"r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["date"] == today:
            marked_today.add(row["employee_id"])

STABILITY_N, THRESHOLD = 7, 55.0
stable_counts = {}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(120,120))

    for (x,y,w,h) in faces:
        face = cv2.resize(gray[y:y+h,x:x+w], (200,200))
        label, confidence = recognizer.predict(face)

        if confidence <= THRESHOLD and label in label_map:
            person = label_map[label]
            cv2.putText(frame,f"{person['id']} {person['name']}",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            stable_counts[label] = stable_counts.get(label,0)+1
            if stable_counts[label] >= STABILITY_N and person["id"] not in marked_today:
                now = datetime.datetime.now()
                with open(CSV_PATH,"a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([today, now.strftime("%H:%M:%S"), person["id"], person["name"]])
                marked_today.add(person["id"])
                print(f"Attendance marked for {person['id']} - {person['name']}")
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,"Unknown",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
