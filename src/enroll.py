import os
import pickle

import cv2
from deepface import DeepFace

# مسار مجلد الصور
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_DIR = os.path.join(BASE_DIR, "Data", "known_people")   # تأكد انها Data أو data حسب فولدراتك
OUTPUT = os.path.join(BASE_DIR, "Data", "embeddings.pkl")

def enroll_known_people():
    embeddings_db = {}

    print("[INFO] Starting enrollment...")
    for filename in os.listdir(KNOWN_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            person_name = os.path.splitext(filename)[0]
            img_path = os.path.join(KNOWN_DIR, filename)

            print(f"[INFO] Processing: {person_name}")

            try:
                # استخراج embedding للوجه من الصورة
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    enforce_detection=True
                )[0]["embedding"]

                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Unable to read image for augmentation")
                flipped_img = cv2.flip(img, 1)

                flipped_embedding = DeepFace.represent(
                    img_path=flipped_img,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    enforce_detection=True
                )[0]["embedding"]

                embeddings_db[person_name] = [embedding, flipped_embedding]
                print(f"[OK] Saved embeddings (original + flipped) for {person_name}")

            except Exception as e:
                print(f"[ERROR] Failed on {filename}: {e}")

    # حفظ النتائج في ملف
    with open(OUTPUT, "wb") as f:
        pickle.dump(embeddings_db, f)

    print("[DONE] Enrollment completed!")
    print(f"[SAVED] Database file: {OUTPUT}")

if __name__ == "__main__":
    enroll_known_people()
# test git change
