import os
import pickle
import shutil
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from tqdm import tqdm
from gfpgan import GFPGANer
import insightface

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "Data", "embeddings.pkl")
NEW_PHOTOS_DIR = os.path.join(BASE_DIR, "Data", "new_photos")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "output")

STRICT_THRESHOLD = 0.45
DOUBT_THRESHOLD = 0.3
QUALITY_GATE_SCORE = 0.6

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Database file not found: {EMBEDDINGS_PATH}")
    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings_db = pickle.load(f)
    print(f"[INFO] Loaded embeddings for {len(embeddings_db)} people")
    return embeddings_db

def ensure_output_dirs(names):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Ensure person-specific folders
    for name in names:
        person_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
    # Ensure Unknown folder for non-matches
    unknown_dir = os.path.join(OUTPUT_DIR, "Unknown")
    os.makedirs(unknown_dir, exist_ok=True)

def cosine_similarity(emb1, emb2):
    return 1 - cosine(emb1, emb2)

def find_best_match(embedding, embeddings_db):
    best_name = None
    best_similarity = -1.0

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
                sim = cosine_similarity(target, saved)
            except Exception as calc_err:
                continue

            if sim > best_similarity:
                best_similarity = sim
                best_name = person_name

    return best_name, best_similarity

def crop_face(img, bbox, margin=0.2):
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = img.shape[:2]
    
    width = x2 - x1
    height = y2 - y1
    
    x1 = max(0, int(x1 - width * margin))
    y1 = max(0, int(y1 - height * margin))
    x2 = min(w, int(x2 + width * margin))
    y2 = min(h, int(y2 + height * margin))
    
    return img[y1:y2, x1:x2]

def restore_face_with_gfpgan(gfpgan_model, face_crop):
    try:
        result = gfpgan_model.enhance(
            face_crop,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5
        )

        if result is None:
            return None

        if isinstance(result, tuple):
            restored_img = result[0] if len(result) > 0 else None
        else:
            restored_img = result

        return restored_img
    except Exception as e:
        return None

def initialize_models():
    print("[INFO] Initializing Dual-Engine InsightFace (buffalo_l) on GPU...")
    
    # Initialize app_small for low-res images (det_size=320)
    print("[INFO] Initializing app_small (det_size=320) for low-res images...")
    try:
        app_small = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider'],
            ctx_id=0
        )
        app_small.prepare(ctx_id=0, det_size=(320, 320))
        print("[OK] app_small loaded on GPU (det_size=(320, 320)).")
    except Exception as e:
        print(f"[WARN] Failed to initialize app_small on GPU: {e}")
        print("[INFO] Falling back to CPU for app_small...")
        try:
            app_small = insightface.app.FaceAnalysis(name='buffalo_l')
            app_small.prepare(ctx_id=-1, det_size=(320, 320))
            print("[OK] app_small loaded on CPU (fallback, det_size=(320, 320)).")
        except Exception as fallback_err:
            print(f"[ERROR] Failed to initialize app_small on CPU: {fallback_err}")
            raise

    # Initialize app_hd for HD/4K images (det_size=640)
    print("[INFO] Initializing app_hd (det_size=640) for HD/4K images...")
    try:
        app_hd = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider'],
            ctx_id=0
        )
        app_hd.prepare(ctx_id=0, det_size=(640, 640))
        print("[OK] app_hd loaded on GPU (det_size=(640, 640)).")
    except Exception as e:
        print(f"[WARN] Failed to initialize app_hd on GPU: {e}")
        print("[INFO] Falling back to CPU for app_hd...")
        try:
            app_hd = insightface.app.FaceAnalysis(name='buffalo_l')
            app_hd.prepare(ctx_id=-1, det_size=(640, 640))
            print("[OK] app_hd loaded on CPU (fallback, det_size=(640, 640)).")
        except Exception as fallback_err:
            print(f"[ERROR] Failed to initialize app_hd on CPU: {fallback_err}")
            raise

    print("[INFO] Loading GFPGAN model...")
    try:
        model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        gfpgan_model = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
        print("[OK] GFPGAN model loaded successfully.")
    except Exception as e:
        print(f"[WARN] Failed to load GFPGAN model: {e}")
        print("[INFO] Continuing without GFPGAN restoration...")
        gfpgan_model = None

    return app_small, app_hd, gfpgan_model

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

    print(f"[INFO] Found {len(images)} images to process")
    print(f"[INFO] Searching for {len(embeddings_db)} known people")
    print(f"[INFO] Thresholds: Strict={STRICT_THRESHOLD}, Doubt={DOUBT_THRESHOLD}, Quality Gate={QUALITY_GATE_SCORE}")
    print(f"[INFO] Dual-Engine: app_small (320x320) for images < 800px, app_hd (640x640) for images >= 800px")

    app_small, app_hd, gfpgan_model = initialize_models()

    stats = {
        'processed': 0,
        'clear_matches': 0,
        'recovered': 0,
        'unknown': 0,
        'no_faces': 0,
        'low_quality_faces': 0,
        'person_counts': {name: 0 for name in embeddings_db.keys()}
    }

    for filename in tqdm(images, desc="Processing images"):
        img_path = os.path.join(NEW_PHOTOS_DIR, filename)

        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Smart routing: Choose app based on image dimensions
            h, w = img.shape[:2]
            max_dim = max(h, w)
            
            if max_dim < 800:
                # Low-res image -> use app_small
                app = app_small
            else:
                # HD/4K image -> use app_hd
                app = app_hd

            faces = app.get(img)
            if not faces:
                stats['no_faces'] += 1
                continue

            stats['processed'] += 1
            image_matched = False

            for face in faces:
                if face.det_score < QUALITY_GATE_SCORE:
                    stats['low_quality_faces'] += 1
                    continue

                embedding = face.embedding
                best_name, best_similarity = find_best_match(embedding, embeddings_db)

                if best_similarity >= STRICT_THRESHOLD:
                    person_dir = os.path.join(OUTPUT_DIR, best_name)
                    os.makedirs(person_dir, exist_ok=True)
                    dest_path = os.path.join(person_dir, filename)
                    
                    if not os.path.exists(dest_path):
                        try:
                            shutil.copy2(img_path, dest_path)
                            stats['person_counts'][best_name] += 1
                            if not image_matched:
                                stats['clear_matches'] += 1
                                image_matched = True
                        except Exception as e:
                            print(f"[ERROR] Failed to copy {filename} to {best_name}: {e}")

                elif best_similarity >= DOUBT_THRESHOLD:
                    # Unsure zone -> Smart Rescue with GFPGAN
                    if gfpgan_model is None:
                        # No restoration available; treat as unknown/ignored
                        continue

                    try:
                        face_crop = crop_face(img, face.bbox)
                        if face_crop.size == 0:
                            continue

                        restored_face = restore_face_with_gfpgan(gfpgan_model, face_crop)
                        if restored_face is None:
                            continue

                        # Re-detect on restored face using the same app that detected it initially
                        restored_faces = app.get(restored_face)
                        if restored_faces:
                            restored_face_obj = max(restored_faces, key=lambda f: f.det_score)
                            if restored_face_obj.det_score >= QUALITY_GATE_SCORE:
                                restored_embedding = restored_face_obj.embedding
                                restored_name, restored_similarity = find_best_match(restored_embedding, embeddings_db)

                                if restored_similarity >= STRICT_THRESHOLD:
                                    person_dir = os.path.join(OUTPUT_DIR, restored_name)
                                    os.makedirs(person_dir, exist_ok=True)
                                    dest_path = os.path.join(person_dir, filename)
                                    
                                    if not os.path.exists(dest_path):
                                        try:
                                            shutil.copy2(img_path, dest_path)
                                            stats['person_counts'][restored_name] += 1
                                            stats['recovered'] += 1
                                            image_matched = True
                                        except Exception as e:
                                            print(f"[ERROR] Failed to copy recovered {filename} to {restored_name}: {e}")

                    except Exception as e:
                        # Any error in rescue path -> just ignore this face
                        continue

                else:
                    # similarity < DOUBT_THRESHOLD -> treat as stranger (ignored)
                    continue

            # If no face in this image produced a match, save to Unknown
            if not image_matched:
                unknown_dir = os.path.join(OUTPUT_DIR, "Unknown")
                dest_path = os.path.join(unknown_dir, filename)
                if not os.path.exists(dest_path):
                    try:
                        shutil.copy2(img_path, dest_path)
                    except Exception as e:
                        print(f"[ERROR] Failed to copy {filename} to Unknown: {e}")
                stats['unknown'] += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            continue

    print("\n" + "="*60)
    print("[FINAL SUMMARY REPORT]")
    print("="*60)
    print(f"Total images processed: {stats['processed']}")
    print(f"Clear matches (>= {STRICT_THRESHOLD}): {stats['clear_matches']}")
    print(f"Recovered via GFPGAN: {stats['recovered']}")
    print(f"Unknown images: {stats['unknown']}")
    print(f"Images with no faces: {stats['no_faces']}")
    print(f"Low quality faces skipped: {stats['low_quality_faces']}")
    print(f"\nTotal matched: {stats['clear_matches'] + stats['recovered']}")
    print(f"Match rate: {(stats['clear_matches'] + stats['recovered']) / max(stats['processed'], 1) * 100:.2f}%")
    print("\n[PERSON BREAKDOWN]")
    for person, count in sorted(stats['person_counts'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {person}: {count} images")
    print("="*60)

if __name__ == "__main__":
    print("[INFO] Starting Smart Pipeline processing...")
    process_new_photos()
    print("[DONE] Processing completed.")
