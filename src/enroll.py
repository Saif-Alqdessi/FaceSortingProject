import os
import pickle
import numpy as np
import cv2
import insightface

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_DIR = os.path.join(BASE_DIR, "Data", "known_people")
OUTPUT = os.path.join(BASE_DIR, "Data", "embeddings.pkl")

def get_largest_face_embedding(app, img):
    if img is None:
        print("[ERROR] Image is None before detection.")
        return None

    faces = app.get(img)
    if not faces:
        return None
    
    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest_face.embedding

def enroll_known_people():
    print("[INFO] Initializing InsightFace model (buffalo_l) on GPU...")
    try:
        app = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider'],
            ctx_id=0
        )
        app.prepare(ctx_id=0, det_size=(320, 320))
        print("[OK] InsightFace model loaded successfully on GPU (providers=['CUDAExecutionProvider'], det_size=(320, 320)).")
    except Exception as e:
        print(f"[ERROR] Failed to initialize InsightFace on GPU: {e}")
        print("[INFO] Falling back to CPU with same det_size (320, 320)...")
        try:
            app = insightface.app.FaceAnalysis(name='buffalo_l')
            app.prepare(ctx_id=-1, det_size=(320, 320))
            print("[OK] InsightFace model loaded on CPU (fallback, det_size=(320, 320)).")
        except Exception as fallback_err:
            print(f"[ERROR] Failed to initialize InsightFace on CPU (fallback): {fallback_err}")
            return

    embeddings_db = {}

    print("[INFO] Starting enrollment...")
    for filename in os.listdir(KNOWN_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            person_name = os.path.splitext(filename)[0]
            img_path = os.path.join(KNOWN_DIR, filename)

            print(f"[INFO] Processing: {person_name}")

            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Unable to read image from disk: {img_path}")
                continue

            embedding = get_largest_face_embedding(app, img)
            if embedding is None:
                print(f"[SKIP] No face detected in {filename}")
                continue

            flipped_img = cv2.flip(img, 1)
            flipped_embedding = get_largest_face_embedding(app, flipped_img)
            if flipped_embedding is None:
                print(f"[SKIP] No face detected in flipped version of {filename}")
                flipped_embedding = embedding

            embeddings_db[person_name] = [embedding, flipped_embedding]
            print(f"[OK] Saved embeddings (original + flipped) for {person_name}")

    with open(OUTPUT, "wb") as f:
        pickle.dump(embeddings_db, f)

    print("[DONE] Enrollment completed!")
    print(f"[SAVED] Database file: {OUTPUT}")

if __name__ == "__main__":
    enroll_known_people()
