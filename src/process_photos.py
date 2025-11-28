print("DEBUG: process_photos.py started, __name__ =", __name__)
import os
import pickle
import shutil
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from tqdm import tqdm

# تحديد المسارات الأساسية (لا تغيّرها لو هيكل المشروع نفس اللي عندك)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "Data", "embeddings.pkl")
NEW_PHOTOS_DIR = os.path.join(BASE_DIR, "Data", "new_photos")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "output")

# ثريشولد المطابقة (Cosine Distance) مناسب لـ ArcFace
THRESHOLD = 0.50  # قيم أقل = تطابق أدق

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

def get_face_embeddings(img_path):
    """
    يرجع قائمة embeddings لكل الوجوه في الصورة
    """
    reps = DeepFace.represent(
        img_path=img_path,
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=True
    )
    if isinstance(reps, list):
        return [np.array(rep["embedding"]) for rep in reps]
    return [np.array(reps["embedding"])]

def find_best_match(embedding, embeddings_db):
    """
    يقارن embedding الجديد مع قاعدة البيانات ويرجع:
    (best_name, best_distance)
    """
    best_name = None
    best_dist = float("inf")

    for person_name, person_emb_list in embeddings_db.items():
        if not isinstance(person_emb_list, (list, tuple)):
            person_emb_iterable = [person_emb_list]
        else:
            person_emb_iterable = person_emb_list

        for stored_emb in person_emb_iterable:
            try:
                target = np.array(embedding).flatten()
                saved = np.array(stored_emb).flatten()
                if target.shape != saved.shape:
                    continue
                dist = cosine(target, saved)
            except Exception as calc_err:
                print(f"[WARN] Cosine calc failed for {person_name}: {calc_err}")
                continue

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

    processed_count = 0
    matched_images = 0
    unknown_images = 0

    for filename in tqdm(images, desc="Processing images"):
        img_path = os.path.join(NEW_PHOTOS_DIR, filename)

        try:
            embeddings = get_face_embeddings(img_path)
        except Exception as e:
            print(f"[SKIP] Could not get face from {filename} -> {e}")
            continue

        if not embeddings:
            print(f"[WARNING] No faces detected in {filename}")
            continue

        processed_count += 1

        matched_people = set()
        closest_candidate = None
        closest_distance = float("inf")
        for emb in embeddings:
            best_name, best_dist = find_best_match(emb, embeddings_db)
            if best_name and best_dist < closest_distance:
                closest_candidate = best_name
                closest_distance = best_dist
            if best_dist <= THRESHOLD:
                matched_people.add(best_name)

        if matched_people:
            import shutil
            for person in matched_people:
                person_dir = os.path.join(OUTPUT_DIR, person)
                os.makedirs(person_dir, exist_ok=True)
                dest_path = os.path.join(person_dir, filename)
                if os.path.exists(dest_path):
                    print(f"[SKIP] {filename} already exists in {person_dir}")
                    continue
                shutil.copy2(img_path, dest_path)
                print(f"  -> Assigned to {person}")
            matched_images += 1
        else:
            # لو ما انطبقت كل الوجوه على أي شخص نخلي الصورة بمجلد Unknown
            unknown_dir = os.path.join(OUTPUT_DIR, "Unknown")
            os.makedirs(unknown_dir, exist_ok=True)
            dest_path = os.path.join(unknown_dir, filename)
            if os.path.exists(dest_path):
                print(f"[SKIP] {filename} already exists in Unknown")
            else:
                shutil.copy2(img_path, dest_path)
                if closest_candidate is not None and closest_distance != float("inf"):
                    print(f"  -> [DEBUG] Marked as Unknown. Closest candidate was: {closest_candidate} with distance: {closest_distance:.4f}")
                else:
                    print("  -> Marked as Unknown (no matches)")
            unknown_images += 1

    print("[DONE] All new photos processed.")
    print(f"[SUMMARY] Total processed: {processed_count}, matched: {matched_images}, unknown: {unknown_images}")

if __name__ == "__main__":
    print("[INFO] Starting processing new photos...")
    process_new_photos()
    print("[DONE] Finished processing new photos.")


