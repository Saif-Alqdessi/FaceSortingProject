# Face Sorting Project

An automated photo sorting project that organizes images by faces using Face Recognition technology.

## 📋 Description

This project automatically sorts photos and organizes them into separate folders based on the person present in each image. The project uses the DeepFace library with the Facenet model to detect and compare faces.

## ✨ Features

- ✅ Register known people from reference photos
- ✅ Automatically sort new photos by faces
- ✅ Place unknown photos in an "Unknown" folder
- ✅ Uses advanced face recognition technology (Facenet + RetinaFace)

## 🛠️ Technologies Used

- **Python 3.x**
- **DeepFace**: Face recognition library
- **Facenet**: Deep learning model for extracting embeddings
- **RetinaFace**: Face detector
- **NumPy**: For numerical operations

## 📁 Project Structure

```
FaceSortingProject/
├── src/
│   ├── enroll.py              # Register known people
│   └── process_photos.py      # Process and sort new photos
├── Data/
│   ├── known_people/          # Reference photos of known people
│   ├── new_photos/            # New photos to be sorted
│   ├── output/                # Photos sorted by person
│   └── embeddings.pkl         # Embeddings database
├── requirements.txt
└── README.md
```

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd FaceSortingProject
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install required libraries

```bash
pip install -r requirements.txt
```

## 📖 Usage

### Step 1: Register Known People

Place reference photos of known people in the `Data/known_people/` folder, naming each image with the person's name (e.g., `Jafar.JPG`, `Saif.JPG`).

Then run:

```bash
python src/enroll.py
```

This will create the `Data/embeddings.pkl` file containing embeddings for all known people.

### Step 2: Sort New Photos

Place new photos to be sorted in the `Data/new_photos/` folder.

Then run:

```bash
python src/process_photos.py
```

Photos will be copied to separate folders in `Data/output/` based on the matched person, or to the `Unknown` folder if no match is found.

## ⚙️ Configuration

You can modify the `THRESHOLD` value in `src/process_photos.py` to change matching sensitivity:

```python
THRESHOLD = 10.0  # Lower values = more precise matching (may miss some matches)
                  # Higher values = more lenient matching (may have some false positives)
```

## 📝 Notes

- The project supports images in formats: `.jpg`, `.jpeg`, `.png`
- Each image in `known_people` should contain only one face
- Images in `new_photos` can contain one or more faces (only the first face will be used)
- On first run, DeepFace models will be downloaded automatically (may take some time)

## 🤝 Contributing

Contributions are welcome! Please open an Issue or Pull Request.

## 📄 License

This project is open source and available for free use.
