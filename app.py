import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import tempfile
import zipfile
import glob
import pandas as pd
import torch
import io

MODEL_PATH = "model.pt" 
CLASS_MAP_OVERRIDE = None  
MAX_SHOW_IMAGES = 20 # for zip

# Functions
def device_str():
    return "0" if torch.cuda.is_available() else "cpu"

def load_yolo_model(path):
    model = YOLO(path)
    return model

def results_to_detections(results):
    res = results
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return []
    try:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)
    except Exception:
        # Fallback robust extraction
        xyxy = np.array(boxes.xyxy)
        confs = np.array(boxes.conf)
        cls_ids = np.array(boxes.cls).astype(int)

    detections = []
    for i in range(len(confs)):
        cid = int(cls_ids[i])
        name = None
        try:
            name = res.names[cid] if hasattr(res, "names") and cid in res.names else None
        except Exception:
            name = None
        if CLASS_MAP_OVERRIDE and cid in CLASS_MAP_OVERRIDE:
            name = CLASS_MAP_OVERRIDE[cid]
        detections.append({
            "class_id": cid,
            "class_name": name,
            "conf": float(confs[i]),
            "xyxy": [float(x) for x in xyxy[i].tolist()]
        })
    return detections

def annotate_from_result(results):
    try:
        annotated = results.plot()  # returns numpy (RGB)
        return annotated
    except Exception:
        try:
            return results.orig_img
        except Exception:
            return None

def infer_on_image(model, pil_img, conf_threshold=0.25, device=None):
    img_np = np.array(pil_img)  # RGB numpy array
    dev = device if device is not None else device_str()

    results = model.predict(source=img_np, device=dev, save=False, verbose=False)
    if not results or len(results) == 0:
        return False, 0.0, [], None, None
    res0 = results[0]
    detections = results_to_detections(res0)
    max_conf = max([d["conf"] for d in detections], default=0.0)
    occupied = any(d["conf"] >= conf_threshold for d in detections)
    annotated = annotate_from_result(res0)
    return occupied, max_conf, detections, annotated, res0


# Streamlit
st.set_page_config(page_title="Free Parking Space Detector", layout="wide")
st.title("🚗 Free Parking Space Detector")

# Sidebar
with st.sidebar:
    st.header("Welcome!")
    st.markdown("This app **utilizes** a pre-trained `model.pt` from the an online dataset.")
    st.markdown("---")
    conf_threshold = st.slider("Detection confidence threshold", 0.0, 1.0, 0.25, 0.01)
    show_boxes_table = st.checkbox("Show detection table per image", value=True)
    show_annotated = st.checkbox("Show annotated images", value=True)
    st.markdown("---")
    st.markdown(f"Device detected: **{'GPU' if torch.cuda.is_available() else 'CPU'}**")

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Default model file not found at `{MODEL_PATH}`. Place your `model.pt` in the same folder as `app.py`.")
    st.stop()

# Load model
@st.cache_resource(show_spinner=False)
def _load_model_cached(path):
    return load_yolo_model(path)

model = _load_model_cached(MODEL_PATH)

# show model task 
try:
    model_task = model.task if hasattr(model, "task") else "detect"
except Exception:
    model_task = "detect"
st.sidebar.write(f"Model task: **{model_task}**")

try:
    names = model.names if hasattr(model, "names") else None
    if names:
        st.sidebar.write(f"Model has {len(names)} class names.")
except Exception:
    names = None

# Main columns
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Input")
    uploaded_image = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    use_cam = st.checkbox("Use webcam (camera)", value=False)
    if use_cam:
        cam_img = st.camera_input("Take a photo")
    else:
        cam_img = None

    uploaded_zip = st.file_uploader("Or upload a ZIP dataset (images inside)", type=["zip"])

with col_right:
    st.subheader("Results")
    result_area = st.empty()

# process image
def process_and_display(pil_img, label=None):
    occupied, max_conf, dets, annotated, raw = infer_on_image(model, pil_img, conf_threshold, device=device_str())
    status = "occupied" if occupied else "empty"
    # header
    header = f"Prediction: **{status}**  —  max_conf: {max_conf:.3f}"
    if label:
        header = f"{label}  |  " + header
    with result_area.container():
        st.markdown(header)
        if show_annotated and annotated is not None:
            st.image(annotated, use_container_width=True)
        if show_boxes_table and dets:
            df = pd.DataFrame(dets)
            # Beautify class_name column if none
            if df["class_name"].isnull().any() and names:
                df["class_name"] = df["class_id"].apply(lambda x: names[int(x)] if int(x) in names else None)
            st.dataframe(df)

    return {"prediction": status, "max_conf": max_conf, "detections": dets, "annotated": annotated}

# Single upload or camera
if cam_img:
    pil = Image.open(io.BytesIO(cam_img.getvalue())).convert("RGB")
    process_and_display(pil, label="webcam")
elif uploaded_image:
    pil = Image.open(io.BytesIO(uploaded_image.read())).convert("RGB")
    process_and_display(pil, label=uploaded_image.name)


# ZIP dataset processing
if uploaded_zip:
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip) as z:
        z.extractall(tmpdir)
    # find images
    files = []
    for root, _, fnames in os.walk(tmpdir):
        for f in fnames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(root, f))
    files = sorted(files)
    st.info(f"Extracted ZIP: {len(files)} images found.")

    if st.button("Run YOLO on ZIP dataset"):
        rows = []
        progress = st.progress(0)
        zip_results = []
        for i, p in enumerate(files):
            pil = Image.open(p).convert("RGB")
            # infer directly (no display)
            occupied, max_conf, dets, annotated, raw = infer_on_image(model, pil, conf_threshold, device=device_str())
            status = "occupied" if occupied else "empty"
            classes = []
            for d in dets:
                cname = d.get("class_name") if d.get("class_name") is not None else str(d.get("class_id"))
                classes.append(str(cname))
            row = {
                "file": os.path.basename(p),
                "prediction": status,
                "max_conf": max_conf,
                "num_detections": len(dets),
                "classes": ",".join(classes) if classes else "",
                "detections": dets,
                "annotated": annotated,
                "src_path": p
            }
            rows.append(row)
            zip_results.append(row)
            progress.progress((i + 1) / len(files))
            if i+1 >= MAX_SHOW_IMAGES:
                st.info(f"Processed first {MAX_SHOW_IMAGES} images. Full CSV available for download.")

        df = pd.DataFrame(rows)

        # store results 
        st.session_state["zip_results"] = zip_results
        st.session_state["zip_idx"] = 0  # start index

        # Summary
        st.markdown("## Dataset Summary")
        if len(df) == 0:
            st.warning("No images processed.")
        else:
            # Group by prediction
            summary = df.groupby("prediction").agg(
                count=("file", "count"),
                avg_conf=("max_conf", "mean"),
                avg_detections=("num_detections", "mean")
            ).reset_index()
            summary["percentage"] = (summary["count"] / len(df) * 100).round(2)
            # Display summary table
            st.markdown("### Occupancy summary")
            st.dataframe(summary)

            # Top detected classes across all images
            all_classes = (
                df["classes"]
                .replace("", np.nan)
                .dropna()
                .str.split(",")
                .explode()
                .reset_index(drop=True)
            )
            if len(all_classes) > 0:
                class_counts = all_classes.value_counts().reset_index()
                class_counts.columns = ["class_name", "count"]
                st.markdown("### Top detected classes")
                st.dataframe(class_counts)
            else:
                st.markdown("### Top detected classes")
                st.write("No detections with class names were found in this dataset.")

        # download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", csv, "results.csv")

    if "zip_results" in st.session_state and st.session_state.get("zip_results"):
        results_list = st.session_state["zip_results"]
        n = len(results_list)
        # ensure index exists
        if "zip_idx" not in st.session_state:
            st.session_state["zip_idx"] = 0

        # Carousel controls
        st.markdown("## Review Detection")
        col_prev, col_idx, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("◀ Previous"):
                st.session_state["zip_idx"] = max(0, st.session_state["zip_idx"] - 1)
        with col_next:
            if st.button("Next ▶"):
                st.session_state["zip_idx"] = min(n - 1, st.session_state["zip_idx"] + 1)

        # slider to jump to any image
        st.session_state["zip_idx"] = st.slider("Jump to image index", 0, n-1, st.session_state["zip_idx"])

        idx = st.session_state["zip_idx"]
        entry = results_list[idx]

        # Show selected annotated image
        st.markdown(f"### [{idx+1}/{n}] {entry['file']}")
        if show_annotated and entry.get("annotated") is not None:
            st.image(entry["annotated"], use_container_width=True)
        else:
            try:
                orig = Image.open(entry["src_path"]).convert("RGB")
                st.image(orig, use_container_width=True)
            except Exception:
                st.write("Annotated image and original not available.")

        # Show per-image summary
        st.markdown("#### Per-image summary")
        per_row = {
            "file": entry["file"],
            "prediction": entry["prediction"],
            "max_conf": entry["max_conf"],
            "num_detections": entry["num_detections"],
            "classes": entry["classes"]
        }
        st.table(pd.DataFrame([per_row]))

        st.markdown("#### Detections")
        if entry["detections"]:
            det_df = pd.DataFrame(entry["detections"])
            # Beautify class_name column if none
            if det_df["class_name"].isnull().any() and names:
                det_df["class_name"] = det_df["class_id"].apply(lambda x: names[int(x)] if int(x) in names else None)
            st.dataframe(det_df)
        else:
            st.write("No detections for this image.")

st.markdown("---")
st.markdown("**Note:** Occupancy rule: image is 'occupied' when any detection has confidence >= threshold.")
