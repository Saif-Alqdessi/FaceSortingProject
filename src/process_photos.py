print("DEBUG: process_photos.py started, __name__ =", __name__)
import os
import pickle
import numpy as np
from deepface import DeepFace

# تحديد المسارات الأساسية (لا تغيّرها لو هيكل المشروع نفس اللي عندك)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "Data", "embeddings.pkl")
NEW_PHOTOS_DIR = os.path.join(BASE_DIR, "Data", "new_photos")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "output")

# ثريشولد المطابقة (ممكن تغييره لاحقاً حسب النتائج)
THRESHOLD = 10.0  # جرّب 0.8 أو 1.0 لو حسيت في أخطاء

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Database file not found: {EMBEDDINGS_PATH}")
    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings_db = pickle.load(f)
    print(f"[INFO] Loaded embeddings for {len(embeddings_db)} people")
    return embeddings_db

def ensure_output_dirs(names):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in names:
        person_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

def get_face_embedding(img_path):
    """
    يرجع الـ embedding للصورة (نفترض صورة شخص واحد في كل صورة كبداية)
    """
    reps = DeepFace.represent(
        img_path=img_path,
        model_name="Facenet",
        detector_backend="retinaface",
        enforce_detection=True
    )
    # DeepFace.represent ترجع list، نأخذ أول وجه
    if isinstance(reps, list):
        return np.array(reps[0]["embedding"])
    else:
        return np.array(reps["embedding"])

def find_best_match(embedding, embeddings_db):
    """
    يقارن embedding الجديد مع قاعدة البيانات ويرجع:
    (best_name, best_distance)
    """
    best_name = None
    best_dist = float("inf")

    for person_name, person_emb in embeddings_db.items():
        dist = np.linalg.norm(embedding - person_emb)
        if dist < best_dist:
            best_dist = dist
            best_name = person_name

    return best_name, best_dist

def process_new_photos():
    embeddings_db = load_embeddings()
    ensure_output_dirs(embeddings_db.keys())

    if not os.path.isdir(NEW_PHOTOS_DIR):
        print(f"[ERROR] New photos folder not found: {NEW_PHOTOS_DIR}")
        return

    images = [f for f in os.listdir(NEW_PHOTOS_DIR)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        print("[WARNING] No images found in new_photos folder.")
        return

    print(f"[INFO] Found {len(images)} images in new_photos")

    for filename in images:
        img_path = os.path.join(NEW_PHOTOS_DIR, filename)
        print(f"[PROCESS] {filename}")

        try:
            emb = get_face_embedding(img_path)
        except Exception as e:
            print(f"[SKIP] Could not get face from {filename} -> {e}")
            continue

        best_name, best_dist = find_best_match(emb, embeddings_db)

        if best_dist <= THRESHOLD:
            # نسخ الصورة لمجلد الشخص
            person_dir = os.path.join(OUTPUT_DIR, best_name)
            os.makedirs(person_dir, exist_ok=True)

            dest_path = os.path.join(person_dir, filename)
            # نستخدم copy بسيط
            import shutil
            shutil.copy2(img_path, dest_path)

            print(f"  -> Assigned to {best_name} (distance={best_dist:.4f})")
        else:
            # لو ما انطبقت على ولا شخص نخليها بمجلد Unknown
            unknown_dir = os.path.join(OUTPUT_DIR, "Unknown")
            os.makedirs(unknown_dir, exist_ok=True)

            import shutil
            shutil.copy2(img_path, os.path.join(unknown_dir, filename))
            print(f"  -> Marked as Unknown (distance={best_dist:.4f})")

    print("[DONE] All new photos processed.")

if __name__ == "__main__":
    print("[INFO] Starting processing new photos...")
    process_new_photos()
    print("[DONE] Finished processing new photos.")


