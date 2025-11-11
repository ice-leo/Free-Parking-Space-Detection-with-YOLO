# Free Parking Space Detection — Streamlit + Ultralytics YOLO

A Streamlit web app for detecting **empty** and **occupied** parking spaces using a **custom YOLO model** (`.pt`).  
The app supports single-image and ZIP dataset predictions, displays annotated results, and generates CSV and summary reports.

Online web app: [Free Parking Space Detection](https://free-parking-space-detection.streamlit.app/)

Training Dataset: [Roboflow](https://universe.roboflow.com/muhammad-syihab-bdynf/parking-space-ipm1b/dataset/3)

---

## 🚀 Features

✅ Uses your **trained YOLO model** (`model.pt`) only — no user uploads for model files  
✅ **Single-image** and **ZIP dataset** prediction modes  
✅ Automatic **annotation preview** for each image  
✅ **Carousel view** with “Next / Previous” buttons to inspect all results  
✅ **Per-image detection table** and **summary statistics**  
✅ **Downloadable CSV report** containing prediction results  
✅ Configurable **confidence threshold**  
✅ Works with CPU or GPU (auto-detects device)

---

## 🗂️ Project Structure

```
parking-space-detector/
├── app.py               # Streamlit app
├── model.pt             # YOLO model
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 💻 Installation & Setup

### 1️⃣ Create a Virtual Environment

```
Windows:
# bash
python -m venv venv
# To activate it:
venv\Scripts\activate

macOS / Linux:
# bash
python3 -m venv venv
# To activate it:
source venv/bin/activate
```
---

### 2️⃣ Install Dependencies

Create a file named **`requirements.txt`** with the following contents:

```text
streamlit>=1.20
ultralytics>=8.0
pillow
opencv-python
pandas
numpy
torch
```

Then install:

```bash
pip install -r requirements.txt
```

> ⚠️ **PyTorch Note:**  
> Install a compatible Torch version for your device.  
> For CPU-only:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

### 3️⃣ Add Your Model

Place your YOLO model in the same directory as `app.py` and rename it to `model.pt`:

```
parking-space-detector/
├── app.py
└── model.pt
```

---

### 4️⃣ Run the App

```bash
streamlit run app.py
```

Then open the local URL (usually `http://localhost:8501`).

---

## 📸 How It Works

1. The app loads your YOLO model (`model.pt`) using Ultralytics.  
2. You can:
   - Upload a **single image** and see detections instantly.
   - Upload a **ZIP dataset** containing multiple images.
3. For each image, the app:
   - Runs inference.
   - Displays annotated detections.
   - Shows a per-image summary table.
4. After ZIP processing:
   - You can **download** a CSV summary.
   - Review all annotated images in a **carousel**.

---

## 📊 CSV & Summary Outputs

Each image result includes:
| Column | Description |
|:--------|:-------------|
| `file` | Image filename |
| `prediction` | `occupied` or `empty` |
| `max_conf` | Maximum detection confidence |
| `num_detections` | Number of detections |
| `classes` | Detected classes (comma-separated) |

The **dataset summary** includes:
- Total number of images processed  
- Number and % of `occupied` vs `empty` images  
- Average confidence  
- Average detections per image  
- Top detected classes  

---

## 🛠️ Troubleshooting

### ❌ `CUDA or Torch errors`
If using GPU, install CUDA-enabled Torch.  
If not, use the CPU-only command above.

### ❌ Blank predictions
The model may not detect anything — try lowering the confidence threshold.

---

## 🧠 How Occupancy Is Determined

An image is marked **`occupied`** if **any object is detected** with confidence ≥ your selected threshold.  
Otherwise, it is marked **`empty`**.

---


## 👩‍💻 Authors

- [Jeremiah Daniel Regalario](https://github.com/jeremiahdanielregalario)
- [Isaiah John Mariano](https://github.com/ice-leo)
- [Meluisa Montealto](https://github.com/melmontealto)

----

