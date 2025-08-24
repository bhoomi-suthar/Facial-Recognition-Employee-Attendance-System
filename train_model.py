import os, json, cv2
import numpy as np

data_dir = "data"
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

faces, labels = [], []
label_map, label_counter = {}, 0

for person in sorted(os.listdir(data_dir)):
    person_path = os.path.join(data_dir, person)
    if not os.path.isdir(person_path) or "_" not in person:
        continue

    emp_id, emp_name = person.split("_", 1)
    label_map[label_counter] = {"id": emp_id, "name": emp_name}

    for img_file in os.listdir(person_path):
        if img_file.lower().endswith((".jpg",".png",".jpeg")):
            img = cv2.imread(os.path.join(person_path, img_file), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (200,200))
            faces.append(img)
            labels.append(label_counter)
    label_counter += 1

faces_np = np.array(faces, dtype="uint8")
labels_np = np.array(labels, dtype="int32")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces_np, labels_np)

recognizer.write(os.path.join(model_dir,"trainer.yml"))
with open(os.path.join(model_dir,"labels.json"),"w") as f:
    json.dump(label_map, f, indent=2)

print("Training complete. Model saved in 'model/'")
