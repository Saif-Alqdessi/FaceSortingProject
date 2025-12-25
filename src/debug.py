import os
import glob

import cv2
import insightface


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_DIR = os.path.join(BASE_DIR, "Data", "known_people")


def find_test_image():
    """
    Prefer 'Adriana Lima.jpg'. If not found, use the first JPG/JPEG/PNG in KNOWN_DIR.
    """
    preferred_name = "Adriana Lima.jpg"
    preferred_path = os.path.join(KNOWN_DIR, preferred_name)

    if os.path.isfile(preferred_path):
        print(f"[INFO] Using preferred image: {preferred_path}")
        return preferred_path

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    for pattern in patterns:
        matches = glob.glob(os.path.join(KNOWN_DIR, pattern))
        if matches:
            print(f"[INFO] Preferred image not found. Using first match: {matches[0]}")
            return matches[0]

    print(f"[ERROR] No images found in {KNOWN_DIR}")
    return None


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Failed to read image: {path}")
        return None
    print(f"[DEBUG] Loaded image {path} with shape {img.shape} (BGR)")
    return img


def run_config(label, det_size, img):
    print(f"\n===== {label} - det_size={det_size} =====")
    try:
        app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"],
            ctx_id=0,
        )
        app.prepare(ctx_id=0, det_size=det_size)
        print("[INFO] Model initialized with CUDAExecutionProvider.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize InsightFace on GPU for {label}: {e}")
        return

    try:
        faces = app.get(img)
        num_faces = len(faces)
        print(f"[RESULT] {label}: detected {num_faces} faces.")
        if num_faces > 0:
            first = faces[0]
            print(f"[RESULT] {label}: first face det_score = {getattr(first, 'det_score', 'N/A')}")
            print(f"[DEBUG] {label}: first face bbox = {getattr(first, 'bbox', 'N/A')}")
        else:
            print(f"[RESULT] {label}: no faces detected.")
    except Exception as e:
        print(f"[ERROR] Exception during detection for {label}: {e}")


def main():
    img_path = find_test_image()
    if not img_path:
        return

    img = load_image(img_path)
    if img is None:
        return

    # Config A: Standard
    run_config("Config A", det_size=(640, 640), img=img)

    # Config B: Small faces
    run_config("Config B", det_size=(320, 320), img=img)

    # Config C: High resolution
    run_config("Config C", det_size=(1280, 1280), img=img)


if __name__ == "__main__":
    main()


