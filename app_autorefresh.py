import streamlit as st
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import pandas as pd
import torch
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2

MODEL_PATH = r"D:\Users\Isaiah\Desktop\Mini Project 2\model.pt" 

# -------------------------------
# Utility Functions
# -------------------------------
def device_str():
    return "0" if torch.cuda.is_available() else "cpu"

def load_yolo_model(path):
    return YOLO(path)

def results_to_detections(results):
    boxes = getattr(results, "boxes", None)
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    detections = []
    for i in range(len(confs)):
        cid = int(cls_ids[i])
        name = results.names.get(cid, f"class_{cid}")
        detections.append({
            "class_id": cid,
            "class_name": name,
            "conf": float(confs[i]),
            "xyxy": [float(x) for x in xyxy[i].tolist()]
        })
    return detections
    
# Annotating the images
def draw_numbered_boxes(pil_img, detections):
    
    img = np.array(pil_img).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Dynamically adjust font size based on image width
    img_h, img_w = img.shape[:2]
    base_font_scale = 2.0
    font_scale = max(0.8, min(2.5, img_w / 1000 * base_font_scale))  # scales with width
    thickness = int(font_scale * 2)
    text_color = (203, 1, 255)  # pink (BGR)
    border_color = (0, 0, 0)      # black

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = map(int, det["xyxy"])
        label = str(idx)

        # Box color: green (empty), red (occupied)
        color = (0, 255, 0) if "empty" in det["class_name"].lower() else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Center text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = int(x1 + (x2 - x1 - text_w) / 2)
        text_y = int(y1 + (y2 - y1 + text_h) / 2)

        # Draw black border first
        cv2.putText(
            img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            border_color,
            thickness + 2,   # slightly thicker for outline
            cv2.LINE_AA
        )

        # Draw pink text on top
        cv2.putText(
            img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def infer_on_image(model, pil_img):
    img_np = np.array(pil_img)
    results = model.predict(source=img_np, device=device_str(), save=False, verbose=False)
    if not results:
        return False, 0.0, [], None

    res0 = results[0]
    detections = results_to_detections(res0)
    annotated = draw_numbered_boxes(pil_img, detections)

    # Determine occupancy based on the most confident detection
    max_conf = max([d["conf"] for d in detections], default=0.0)
    occupied = len(detections) > 0  # occupied if any detection exists

    return occupied, max_conf, detections, annotated

# -------------------------------
# Streamlit Setup
# -------------------------------
st.set_page_config(page_title="Smart Parking Detector", layout="wide")

@st.cache_resource(show_spinner=False)
def _load_model_cached(path):
    return load_yolo_model(path)

model = _load_model_cached(MODEL_PATH)

# Initialize session state
if "parking_data" not in st.session_state:
    st.session_state["parking_data"] = {
        "Lot 1": {"spaces": [], "reservations": {}, "last_detection": None},
        "Lot 2": {"spaces": [], "reservations": {}, "last_detection": None},
        "Lot 3": {"spaces": [], "reservations": {}, "last_detection": None}
    }

if "detection_results" not in st.session_state:
    st.session_state["detection_results"] = {
        "Lot 1": {"occupied": False, "max_conf": 0.0, "annotated": None, "timestamp": None},
        "Lot 2": {"occupied": False, "max_conf": 0.0, "annotated": None, "timestamp": None},
        "Lot 3": {"occupied": False, "max_conf": 0.0, "annotated": None, "timestamp": None}
    }

tab1, tab2 = st.tabs(["🧠 Detection", "🅿️ Reservation"])

# -------------------------------
# TAB 1: Detection
# -------------------------------
with tab1:
    st.title("🚗 Free Parking Space Detector")

    cols = st.columns(3)
    for i, col in enumerate(cols, start=1):
        lot_name = f"Lot {i}"
        with col:
            st.subheader(lot_name)
            input_mode = st.radio(
                f"Choose input method for {lot_name}:",
                ["Upload Image", "Use Webcam"],
                key=f"input_mode_{i}"
            )

            pil_img = None
            if input_mode == "Upload Image":
                uploaded_img = st.file_uploader(
                    f"Upload image for {lot_name}",
                    type=["jpg", "jpeg", "png"],
                    key=f"file_{i}"
                )
                if uploaded_img:
                    pil_img = Image.open(io.BytesIO(uploaded_img.read())).convert("RGB")

            else:
                cam_img = st.camera_input(f"Take photo for {lot_name}", key=f"cam_{i}")
                if cam_img:
                    pil_img = Image.open(io.BytesIO(cam_img.getvalue())).convert("RGB")

            if pil_img is not None and st.button(f"Run Detection for {lot_name}", key=f"detect_{i}"):
                with st.spinner(f"Running YOLO detection for {lot_name}..."):
                    occupied, max_conf, detections, annotated = infer_on_image(model, pil_img)

                    # Get previous reservations
                    prev_spaces = st.session_state["parking_data"][lot_name]["spaces"]
                    reservations = st.session_state["parking_data"][lot_name]["reservations"]

                    # Count all detected parking-related objects (cars, empty spaces, etc.)
                    if detections:
                        # Each detection corresponds to a space (either empty or occupied)
                        TOTAL_SPACES = len(detections)
                    else:
                        # If nothing detected, keep previous known count (if exists) or default to 0
                        prev_spaces = st.session_state["parking_data"][lot_name]["spaces"]
                        TOTAL_SPACES = len(prev_spaces) if prev_spaces else 0

                    new_spaces = []
                    reservations = st.session_state["parking_data"][lot_name]["reservations"]

                    # ------------------------------
                    # Process detected spaces
                    # ------------------------------
                    for idx, det in enumerate(detections, start=1):
                        # Determine status based on detection class name
                        cname = det["class_name"].lower()
                        if "empty" in cname or "free" in cname:
                            status = "empty"
                        else:
                            status = "occupied"  # assume any other detection = vehicle
                        
                        # Reservations override detection
                        if idx in reservations:
                            status = "reserved"

                        new_spaces.append({
                            "id": idx,
                            "status": status,
                            "conf": det["conf"],
                            "class_name": det["class_name"]
                        })

                    # ------------------------------
                    # Fill remaining spaces if total increased or none detected
                    # ------------------------------
                    for idx in range(len(new_spaces) + 1, TOTAL_SPACES + 1):
                        if idx in reservations:
                            status = "reserved"
                        else:
                            status = "empty"
                        new_spaces.append({
                            "id": idx,
                            "status": status,
                            "conf": 0.0,
                            "class_name": None
                        })

                    # ------------------------------
                    # Store results in session state
                    # ------------------------------
                    st.session_state["parking_data"][lot_name]["spaces"] = new_spaces
                    st.session_state["parking_data"][lot_name]["last_detection"] = datetime.now()
                    st.session_state["detection_results"][lot_name] = {
                        "occupied": occupied,
                        "max_conf": max_conf,
                        "annotated": annotated,
                        "timestamp": datetime.now()
                    }
                    st.session_state["parking_data"][lot_name]["annotated_image"] = annotated

            # Always show the latest annotated result
            result = st.session_state["detection_results"][lot_name]
            if result["annotated"] is not None:
                timestamp_str = f" (Detected: {result['timestamp'].strftime('%H:%M:%S')})"
                st.image(
                    result["annotated"],
                    caption=f"{lot_name}: {'Occupied' if result['occupied'] else 'Empty'} "
                            f"(Most confident: {result['max_conf']:.2f}){timestamp_str}"
                )
                
# -------------------------------
# TAB 2: Reservation & Dashboard
# -------------------------------
st_autorefresh(interval=1000, key="refresh_timer")
with tab2:
    st.title("🅿️ Parking Reservation Dashboard")

    # Update reservation timers
    for lot_name, data in st.session_state["parking_data"].items():
        for space in data["spaces"]:
            if space["status"] == "reserved":
                res_info = data["reservations"].get(space["id"])
                if res_info:
                    remaining = (res_info["end_time"] - datetime.now()).total_seconds()
                    if remaining <= 0:
                        # Expired reservation → set to empty
                        space["status"] = "empty"
                        del data["reservations"][space["id"]]

    # Compute per-lot summaries from the current spaces list
    summaries = []
    for lot_name, data in st.session_state["parking_data"].items():
        spaces = data["spaces"]
        if spaces:
            # Count by status
            occupied = sum(1 for s in spaces if s["status"] == "occupied")
            reserved = sum(1 for s in spaces if s["status"] == "reserved")
            empty = sum(1 for s in spaces if s["status"] == "empty")

            # Get detection info for caption
            detection_result = st.session_state["detection_results"].get(lot_name, {})
            confidence = detection_result.get("max_conf", 0.0)
            detection_status = "Occupied" if occupied > 0 else "Empty"

            summaries.append((lot_name, occupied, reserved, empty, detection_status, confidence))
        else:
            summaries.append((lot_name, 0, 0, 0, "No data", 0.0))

    # Render doughnut charts
    cols = st.columns(3)
    for i, (lot_name, occ, res, emp, detection_status, confidence) in enumerate(summaries):
        fig = go.Figure(data=[go.Pie(
            labels=["Occupied", "Reserved", "Empty"],
            values=[occ, res, emp],
            hole=0.5,
            marker_colors=['#EF553B', '#FFA15A', '#00CC96']
        )])
        fig.update_layout(
            title_text=f"{lot_name}<br><sub>{detection_status} (max conf: {confidence:.2f})</sub>",
            showlegend=True
        )
        cols[i].plotly_chart(fig, use_container_width=True)

    selected_lot = st.selectbox("Select a parking lot:", list(st.session_state["parking_data"].keys()))
    spaces = st.session_state["parking_data"][selected_lot]["spaces"]

    if not spaces:
        st.warning("No detection data available yet. Please run detection first.")
    else:
        st.subheader(f"Parking Spaces — {selected_lot}")

        detection_info = st.session_state["detection_results"][selected_lot]
        if detection_info["annotated"] is not None:
            st.info(f"Latest detection: **{'Occupied' if detection_info['occupied'] else 'Empty'}**, "
                    f"most confident: {detection_info['max_conf']:.2f}")
            
        # Display Image
        lot_name = selected_lot  # assuming 'selected_lot' holds the currently selected parking lot name
        if lot_name in st.session_state["parking_data"]:
            lot_data = st.session_state["parking_data"][lot_name]
            if lot_data.get("annotated_image") is not None:
                st.image(
                    lot_data["annotated_image"],
                    caption=f"Detection Image - {lot_name}",
                    width=400
                )
            else:
                st.warning(f"No uploaded detection image found for {lot_name}.")
        else:
            st.warning(f"Lot '{lot_name}' not found in session data.")

        # Reservation (One reservation per user)
        user_has_reservation = False
        reserved_space_info = None
        for lot_name, data in st.session_state["parking_data"].items():
            for sid, res_info in data["reservations"].items():
                # If a space is reserved, we assume it's reserved by "this user/session"
                user_has_reservation = True
                reserved_space_info = (lot_name, sid)
                break
            if user_has_reservation:
                break

        cols = st.columns(4)
        for i, space in enumerate(spaces):
            status = space["status"]
            # Create clear, consistent labels
            if status == "occupied":
                label = f"Space {space['id']} - 🚗 Occupied"
                if space.get("class_name") and space["conf"] > 0:
                    label += f" (conf: {space['conf']:.2f})"
            elif status == "reserved":
                label = f"Space {space['id']} - 📋 Reserved"
            else:  # empty
                label = f"Space {space['id']} - ✅ Empty"

            # Timer for reserved spots
            if status == "reserved":
                end_time = st.session_state["parking_data"][selected_lot]["reservations"].get(space["id"], {}).get("end_time")
                if end_time:
                    remaining = (end_time - datetime.now()).total_seconds()
                    if remaining <= 0:
                        space["status"] = "empty"
                        del st.session_state["parking_data"][selected_lot]["reservations"][space["id"]]
                    else:
                        mins, secs = divmod(int(remaining), 60)
                        label += f" ⏱️ {mins:02d}:{secs:02d}"

            with cols[i % 4]:
                if st.button(label, key=f"{selected_lot}_{space['id']}"):
                    if status == "empty":
                        if user_has_reservation:
                            st.warning(f"You already reserved Space {reserved_space_info[1]} in {reserved_space_info[0]}. Cannot reserve another slot.")
                        else:
                            st.session_state["confirm_reservation"] = (selected_lot, space["id"])
                    elif status == "reserved":
                        st.session_state["selected_reserved"] = (selected_lot, space["id"])
                    elif status == "occupied":
                        st.info("This space is already occupied.")

        # Handle reservation confirmation
        if "confirm_reservation" in st.session_state:
            lot, sid = st.session_state["confirm_reservation"]
            st.warning(f"Reserve Space {sid} in {lot}?")
            coly, coln = st.columns(2)
            with coly:
                if st.button("✅ Yes"):
                    st.session_state["parking_data"][lot]["spaces"][sid - 1]["status"] = "reserved"
                    st.session_state["parking_data"][lot]["reservations"][sid] = {
                        "end_time": datetime.now() + timedelta(minutes=15)
                    }
                    st.session_state["selected_reserved"] = (lot, sid)
                    del st.session_state["confirm_reservation"]
                    st.success("Reservation confirmed!")
                    st.rerun() 
            with coln:
                if st.button("❌ No"):
                    del st.session_state["confirm_reservation"]

        # Handle actions for reserved
        if "selected_reserved" in st.session_state:
            lot, sid = st.session_state["selected_reserved"]
            st.info(f"Space {sid} in {lot} is reserved.")
            colp, colc = st.columns(2)
            with colp:
                if st.button("🚘 Parked"):
                    st.session_state["parking_data"][lot]["spaces"][sid - 1]["status"] = "occupied"
                    del st.session_state["parking_data"][lot]["reservations"][sid]
                    del st.session_state["selected_reserved"]
                    st.rerun()
            with colc:
                if st.button("❌ Cancel"):
                    st.session_state["parking_data"][lot]["spaces"][sid - 1]["status"] = "empty"
                    del st.session_state["parking_data"][lot]["reservations"][sid]
                    del st.session_state["selected_reserved"]
                    st.rerun()

st.markdown("---")
st.caption("Developed by JD, Mel, Isaiah — Smart Parking Detector and Reservation System 🚗")