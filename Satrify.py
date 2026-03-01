import sys
import cv2
import numpy as np
import urllib.request
import os
import subprocess
import json
import logging
import traceback
import tempfile
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSlider,
    QProgressBar,
    QMessageBox,
    QFrame,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QSizePolicy,
    QGroupBox,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QPoint, QRect
from PyQt6.QtGui import QIcon, QImage, QPixmap, QPalette, QColor, QPainter, QPen
import imageio_ffmpeg
import time

# --- Global Cross-Platform Helper Functions ---
def get_real_home():
    return os.environ.get("SNAP_REAL_HOME", os.path.expanduser("~"))


def get_app_data_dir():
    # الحل الجذري: استخدام المسار الرسمي والمصرح به داخل بيئة Snap
    if "SNAP_USER_DATA" in os.environ:
        return os.environ["SNAP_USER_DATA"]
    # في حالة ويندوز أو ماك، ننشئ مجلد بيانات عادي
    data_dir = os.path.join(os.path.expanduser("~"), ".satrify_data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_standard_output_path(filename="processed_video.mp4"):
    base_dir = Path(get_real_home()) / "Videos" / "Satrify_Output"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / filename)


def get_temp_video_path():
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, "temp_satrify_no_audio.mp4")


# --- Logging Setup ---
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SatrifyLogger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

try:
    # حفظ ملف اللوج في المسار المصرح به أمنياً
    log_path = os.path.join(get_app_data_dir(), "satrify_log.txt")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Failed to setup file logger: {e}")

logger.info("=========================================")
logger.info("Satrify Application Started")
logger.info("=========================================")


# --- Smart Display Widget ---
class VideoDisplayLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background-color: #000000; color: #888; border: 1px solid #444; border-radius: 8px; font-size: 18px;"
        )
        self.begin = QPoint()
        self.end = QPoint()
        self.is_drawing = False
        self.drawing_enabled = False
        self.roi_rect = None
        self.video_orig_size = (0, 0)

    def get_pixmap_rect(self):
        if not self.pixmap():
            return QRect()
        pw, ph = self.pixmap().width(), self.pixmap().height()
        lw, lh = self.width(), self.height()
        return QRect((lw - pw) // 2, (lh - ph) // 2, pw, ph)

    def mousePressEvent(self, event):
        if self.drawing_enabled and self.pixmap():
            self.begin = self.end = event.pos()
            self.is_drawing = True
            self.roi_rect = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing_enabled and self.is_drawing:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_enabled and self.is_drawing:
            self.end = event.pos()
            self.is_drawing = False
            self.roi_rect = QRect(self.begin, self.end).normalized()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing_enabled and self.pixmap():
            painter = QPainter(self)
            painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))
            if self.is_drawing:
                painter.drawRect(QRect(self.begin, self.end).normalized())
            elif self.roi_rect:
                painter.drawRect(self.roi_rect)

    def get_video_roi(self):
        if not self.roi_rect or not self.pixmap() or self.video_orig_size == (0, 0):
            return None
        pr = self.get_pixmap_rect()
        intersected = self.roi_rect.intersected(pr)
        if intersected.isEmpty():
            return None

        orig_w, orig_h = self.video_orig_size
        scale_x, scale_y = orig_w / pr.width(), orig_h / pr.height()
        return (
            int((intersected.x() - pr.x()) * scale_x),
            int((intersected.y() - pr.y()) * scale_y),
            int(intersected.width() * scale_x),
            int(intersected.height() * scale_y),
        )


# --- Video Processing Engine ---
class VideoProcessor(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    frame_update = pyqtSignal(np.ndarray)
    frame_idx_update = pyqtSignal(int)

    def __init__(
        self,
        input_path,
        output_path,
        conf_thresh,
        memory_frames,
        blocks_percentage,
        padding,
        mode="auto",
        manual_bbox=None,
        start_frame_idx=0,
        shape="square",
        ai_model="caffe",
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.conf_thresh = conf_thresh
        self.memory_frames = memory_frames
        self.blocks_percentage = blocks_percentage
        self.padding = padding
        self.mode = mode
        self.manual_bbox = manual_bbox
        self.start_frame_idx = start_frame_idx
        self.shape = shape
        self.ai_model = ai_model
        self.show_compare = False
        self._is_running = True
        self._is_paused = False
        logger.info(f"Initialized VideoProcessor: mode={mode}, model={ai_model}")

    def stop(self):
        self._is_running = False
    def toggle_pause(self):
        self._is_paused = not getattr(self, "_is_paused", False)
        return self._is_paused
    def pixelate_face(self, image, block_percentage):
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return image
        block_size = max(1, int(w * (block_percentage / 100.0)))
        x_steps, y_steps = max(1, w // block_size), max(1, h // block_size)
        small = cv2.resize(image, (x_steps, y_steps), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def download_model(self, url, dest_path):
        logger.info(f"Downloading model from: {url}")
        urllib.request.urlretrieve(url, dest_path)

    def run(self):
        try:
            logger.info("Opening video file...")
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(
                f"Video Info: {total_frames} frames, {width}x{height}, {fps} FPS"
            )

            temp_video = get_temp_video_path()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

            net = None
            haar_cascade = None

            if self.mode in ["auto", "hybrid"]:
                # --- [التعديل هنا] استخدام المجلد الآمن أمنياً للموديلات ---
                models_dir = os.path.join(get_app_data_dir(), "models")
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                    logger.info("Created 'models' directory.")

                if self.ai_model == "caffe":
                    prototxt_path = os.path.join(models_dir, "deploy.prototxt")
                    model_path = os.path.join(
                        models_dir, "res10_300x300_ssd_iter_140000.caffemodel"
                    )
                    if not os.path.isfile(prototxt_path) or not os.path.isfile(
                        model_path
                    ):
                        self.status_update.emit("Loading AI models (Caffe)...")
                        self.download_model(
                            "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
                            prototxt_path,
                        )
                        self.download_model(
                            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                            model_path,
                        )
                    logger.info("Loading Caffe network...")
                    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

                elif self.ai_model == "tf":
                    pbtxt_path = os.path.join(models_dir, "opencv_face_detector.pbtxt")
                    pb_path = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
                    if not os.path.isfile(pbtxt_path) or not os.path.isfile(pb_path):
                        self.status_update.emit("Loading AI models (TensorFlow)...")
                        self.download_model(
                            "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/opencv_face_detector.pbtxt",
                            pbtxt_path,
                        )
                        self.download_model(
                            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb",
                            pb_path,
                        )
                    logger.info("Loading TensorFlow network...")
                    net = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)

                elif self.ai_model == "haar":
                    self.status_update.emit("Loading AI models (Haar Cascade)...")
                    cascade_path = (
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    logger.info(f"Loading Haar Cascade from: {cascade_path}")
                    haar_cascade = cv2.CascadeClassifier(cascade_path)
                    if haar_cascade.empty():
                        raise Exception("Failed to load Haar Cascade XML.")

            if self.mode in ["manual", "hybrid"]:
                self.status_update.emit("Initializing smart tracker (CSRT)...")
                logger.info("Initializing CSRT Tracker...")
                tracker = cv2.TrackerCSRT_create()
                tracker_initialized = False

            self.status_update.emit("Processing...")
            logger.info("Started frame processing loop.")
            last_known_faces = []
            frames_missing = 0
            current_frame_idx = 0

            while cap.isOpened() and self._is_running:
                if getattr(self, "_is_paused", False):
                    self.status_update.emit("Paused...")
                    while self._is_paused and self._is_running:
                        time.sleep(0.1)  # تنويم المعالج أجزاء من الثانية عشان ميهنجش
                    if not self._is_running:
                        break  # لو المستخدم داس إيقاف نهائي وهو في وضع الإيقاف المؤقت
                    self.status_update.emit("Processing...")
                success, frame = cap.read()
                if not success:
                    break

                original_frame = frame.copy()

                h_f, w_f = frame.shape[:2]
                current_faces = []
                ai_faces = []
                tracker_box = None

                if self.mode in ["auto", "hybrid"]:
                    if self.ai_model in ["caffe", "tf"]:
                        blob = cv2.dnn.blobFromImage(
                            cv2.resize(frame, (300, 300)),
                            1.0,
                            (300, 300),
                            (104.0, 177.0, 123.0),
                        )
                        net.setInput(blob)
                        detections = net.forward()

                        for i in range(0, detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            if confidence > self.conf_thresh:
                                box = detections[0, 0, i, 3:7] * np.array(
                                    [w_f, h_f, w_f, h_f]
                                )
                                (startX, startY, endX, endY) = box.astype("int")
                                if (endX - startX) < (w_f * 0.8) and (endY - startY) < (
                                    h_f * 0.8
                                ):
                                    ai_faces.append((startX, startY, endX, endY))

                    elif self.ai_model == "haar":
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        neighbors = max(3, int(self.conf_thresh * 10))
                        faces = haar_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=neighbors,
                            minSize=(30, 30),
                        )
                        for x, y, w_b, h_b in faces:
                            ai_faces.append((x, y, x + w_b, y + h_b))

                if self.mode in ["manual", "hybrid"]:
                    if current_frame_idx < self.start_frame_idx:
                        pass
                    elif current_frame_idx == self.start_frame_idx and self.manual_bbox:
                        tracker.init(frame, self.manual_bbox)
                        tracker_initialized = True
                        (x, y, w_b, h_b) = [int(v) for v in self.manual_bbox]
                        tracker_box = (x, y, x + w_b, y + h_b)
                    elif tracker_initialized:
                        success_track, box = tracker.update(frame)
                        if success_track:
                            (x, y, w_b, h_b) = [int(v) for v in box]
                            tracker_box = (x, y, x + w_b, y + h_b)

                if self.mode == "auto":
                    for startX, startY, endX, endY in ai_faces:
                        pad_w = int((endX - startX) * (self.padding / 200.0))
                        pad_h = int((endY - startY) * (self.padding / 200.0))
                        current_faces.append(
                            (
                                max(0, startX - pad_w),
                                max(0, startY - pad_h),
                                min(w_f, endX + pad_w),
                                min(h_f, endY + pad_h),
                            )
                        )

                    if len(current_faces) > 0:
                        last_known_faces = current_faces
                        frames_missing = 0
                    else:
                        frames_missing += 1
                        if frames_missing <= self.memory_frames:
                            current_faces = last_known_faces
                        else:
                            last_known_faces = []
                            current_faces = []

                elif self.mode == "manual" and tracker_box:
                    (startX, startY, endX, endY) = tracker_box
                    pad_w = int((endX - startX) * (self.padding / 200.0))
                    pad_h = int((endY - startY) * (self.padding / 200.0))
                    current_faces.append(
                        (
                            max(0, startX - pad_w),
                            max(0, startY - pad_h),
                            min(w_f, endX + pad_w),
                            min(h_f, endY + pad_h),
                        )
                    )

                elif self.mode == "hybrid" and tracker_box:
                    (t_startX, t_startY, t_endX, t_endY) = tracker_box
                    final_startX, final_startY = t_startX, t_startY
                    final_endX, final_endY = t_endX, t_endY

                    for a_startX, a_startY, a_endX, a_endY in ai_faces:
                        intersect = not (
                            t_endX < a_startX
                            or t_startX > a_endX
                            or t_endY < a_startY
                            or t_startY > a_endY
                        )
                        if intersect:
                            final_startX = min(final_startX, a_startX)
                            final_startY = min(final_startY, a_startY)
                            final_endX = max(final_endX, a_endX)
                            final_endY = max(final_endY, a_endY)

                    pad_w = int((final_endX - final_startX) * (self.padding / 200.0))
                    pad_h = int((final_endY - final_startY) * (self.padding / 200.0))
                    current_faces.append(
                        (
                            max(0, final_startX - pad_w),
                            max(0, final_startY - pad_h),
                            min(w_f, final_endX + pad_w),
                            min(h_f, final_endY + pad_h),
                        )
                    )

                for startX, startY, endX, endY in current_faces:
                    if startX < endX and startY < endY:
                        face_region = frame[startY:endY, startX:endX]
                        pixelated_face = self.pixelate_face(
                            face_region, self.blocks_percentage
                        )

                        if self.shape == "circle":
                            fh, fw = face_region.shape[:2]
                            mask = np.zeros((fh, fw), dtype="uint8")
                            center = (fw // 2, fh // 2)
                            axes = (fw // 2, fh // 2)
                            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                            frame[startY:endY, startX:endX] = np.where(
                                mask[..., None] == 255, pixelated_face, face_region
                            )
                        else:
                            frame[startY:endY, startX:endX] = pixelated_face

                out.write(frame)

                if current_frame_idx % 3 == 0:
                    if self.show_compare:
                        h_c, w_c = frame.shape[:2]
                        if w_c >= h_c:
                            composite = np.vstack((frame, original_frame))
                        else:
                            composite = np.hstack((original_frame, frame))
                        self.frame_update.emit(composite)
                    else:
                        self.frame_update.emit(frame)
                    self.frame_idx_update.emit(current_frame_idx)

                current_frame_idx += 1
                if total_frames > 0:
                    self.progress_update.emit(
                        int((current_frame_idx / total_frames) * 100)
                    )

            cap.release()
            out.release()
            logger.info("Video processing loop completed.")

            if not self._is_running:
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                self.finished.emit(False, "Processing stopped by user.")
                return

            self.status_update.emit("Merging audio...")
            logger.info("Merging audio using FFmpeg...")

            # [الحل السحري الشامل]: استدعاء النسخة المدمجة والمستقلة من ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            
            command = [
                ffmpeg_path, "-y", "-i", temp_video, "-i", self.input_path,
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", self.output_path
            ]
            
            try:
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                logger.info("FFmpeg merge successful.")
                self.finished.emit(True, "Done successfully! Video saved.")
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"FFmpeg audio merge failed: {e}. Saving video without audio track."
                )
                shutil.move(temp_video, self.output_path)
                self.finished.emit(True, "Video saved (without audio track).")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error during processing: {error_msg}")
            logger.error(traceback.format_exc())
            self.finished.emit(False, f"Error: {error_msg}")


# --- Main Application GUI ---
class FaceBlurApp(QWidget):
    def __init__(self):
        super().__init__()
        self.input_file = ""
        self.output_file = ""
        self.preview_cap = None
        self.current_preview_frame = None
        self.video_fps = 30
        self.settings = QSettings("FaceBlurTools", "FaceBlurApp")
        self.available_langs = {}
        self.load_locales()

        self.current_lang_name = str(
            self.settings.value("language", "English (Default)")
        )
        if (
            self.current_lang_name not in self.available_langs
            and self.current_lang_name != "English (Default)"
        ):
            self.current_lang_name = "English (Default)"

        self.initUI()
        self.apply_language()

    def load_locales(self):
        locales_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "locales"
        )
        if os.path.exists(locales_dir):
            for filename in os.listdir(locales_dir):
                if filename.endswith(".json"):
                    try:
                        with open(
                            os.path.join(locales_dir, filename), "r", encoding="utf-8"
                        ) as f:
                            data = json.load(f)
                            if "lang" in data:
                                self.available_langs[data["lang"]] = data
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {e}")

    def tr(self, text):
        return (
            self.available_langs.get(self.current_lang_name, {})
            .get("translations", {})
            .get(text, text)
        )

    def create_tooltip_icon(self):
        lbl = QLabel("?")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFixedSize(18, 18)
        lbl.setStyleSheet(
            "QLabel { background-color: #555; color: #fff; border-radius: 9px; font-size: 11px; font-weight: bold; margin: 0px; } QLabel:hover { background-color: #2a82da; }"
        )
        return lbl

    def initUI(self):
        self.resize(1150, 750)
        self.setStyleSheet("font-size: 14px;")
        self.main_layout = QHBoxLayout()

        # Control Panel
        control_panel = QWidget()
        control_panel.setMaximumWidth(450)
        self.control_layout = QVBoxLayout(control_panel)
        self.control_layout.setContentsMargins(15, 0, 15, 0)
        self.control_layout.setSpacing(10)

        # Header Row (Logo + Title + Language)
        header_layout = QHBoxLayout()
        self.lbl_logo = QLabel()
        icon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "satrify.png"
        )
        if os.path.exists(icon_path):
            self.lbl_logo.setPixmap(
                QPixmap(icon_path).scaled(
                    35,
                    35,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        self.lbl_title_text = QLabel("Satrify Pro")
        self.lbl_title_text.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #4da6ff;"
        )

        self.combo_lang = QComboBox()
        self.combo_lang.addItem("English (Default)")
        for lang in self.available_langs.keys():
            if lang != "English (Default)":
                self.combo_lang.addItem(lang)

        idx = self.combo_lang.findText(self.current_lang_name)
        if idx >= 0:
            self.combo_lang.setCurrentIndex(idx)
        self.combo_lang.currentTextChanged.connect(self.change_language)

        header_layout.addWidget(self.lbl_logo)
        header_layout.addWidget(self.lbl_title_text)
        header_layout.addStretch()
        header_layout.addWidget(self.combo_lang)
        self.control_layout.addLayout(header_layout)

        # File Buttons
        files_btn_layout = QHBoxLayout()
        self.btn_input = QPushButton()
        self.btn_output = QPushButton()
        self.btn_input.setMinimumHeight(40)
        self.btn_output.setMinimumHeight(40)
        self.btn_input.clicked.connect(self.select_input)
        self.btn_output.clicked.connect(self.select_output)
        files_btn_layout.addWidget(self.btn_input)
        files_btn_layout.addWidget(self.btn_output)
        self.control_layout.addLayout(files_btn_layout)

        # File Labels
        files_lbl_layout = QHBoxLayout()
        self.lbl_input = QLabel()
        self.lbl_output = QLabel()
        self.lbl_input.setStyleSheet("color: #aaa; font-size: 12px;")
        self.lbl_output.setStyleSheet("color: #aaa; font-size: 12px;")
        files_lbl_layout.addWidget(self.lbl_input)
        files_lbl_layout.addWidget(self.lbl_output)
        self.control_layout.addLayout(files_lbl_layout)

        # Mode Selection
        self.grp_mode = QGroupBox()
        mode_btn_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.radio_auto = QRadioButton()
        self.radio_manual = QRadioButton()
        self.radio_hybrid = QRadioButton()
        self.radio_auto.setChecked(True)

        for rb in (self.radio_auto, self.radio_manual, self.radio_hybrid):
            self.mode_group.addButton(rb)
            mode_btn_layout.addWidget(rb)

        self.grp_mode.setLayout(mode_btn_layout)
        self.control_layout.addWidget(self.grp_mode)
        self.radio_manual.toggled.connect(self.toggle_drawing_mode)
        self.radio_hybrid.toggled.connect(self.toggle_drawing_mode)

        # Shape Selection
        self.grp_shape = QGroupBox()
        shape_layout = QHBoxLayout()
        self.shape_group = QButtonGroup(self)
        self.radio_square = QRadioButton()
        self.radio_circle = QRadioButton()

        if str(self.settings.value("shape", "square")) == "circle":
            self.radio_circle.setChecked(True)
        else:
            self.radio_square.setChecked(True)

        for rb in (self.radio_square, self.radio_circle):
            self.shape_group.addButton(rb)
            shape_layout.addWidget(rb)

        self.grp_shape.setLayout(shape_layout)
        self.control_layout.addWidget(self.grp_shape)

        # Settings
        self.grp_settings = QGroupBox()
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(5)

        h_ai = QHBoxLayout()
        self.lbl_ai = QLabel()
        self.tip_ai = self.create_tooltip_icon()
        self.combo_ai = QComboBox()
        self.combo_ai.addItems(
            [
                "DNN Caffe (Balanced)",
                "DNN TensorFlow (Accurate)",
                "Haar Cascade (Fast/Frontal)",
            ]
        )

        saved_ai = str(self.settings.value("ai_model", "caffe"))
        idx_ai = self.combo_ai.findData(saved_ai)
        if idx_ai >= 0:
            self.combo_ai.setCurrentIndex(idx_ai)
        self.combo_ai.currentIndexChanged.connect(
            lambda: self.settings.setValue("ai_model", self.combo_ai.currentData())
        )

        h_ai.addWidget(self.lbl_ai)
        h_ai.addWidget(self.tip_ai)
        h_ai.addStretch()
        h_ai.addWidget(self.combo_ai)
        settings_layout.addLayout(h_ai)

        self.slider_conf = self.create_slider(
            10, 100, int(self.settings.value("conf", 30)), self.update_labels
        )
        self.lbl_conf, self.tip_conf = QLabel(), self.create_tooltip_icon()
        self.add_slider_row(
            settings_layout, self.lbl_conf, self.tip_conf, self.slider_conf
        )

        self.slider_memory = self.create_slider(
            0, 30, int(self.settings.value("memory", 5)), self.update_labels
        )
        self.lbl_memory, self.tip_mem = QLabel(), self.create_tooltip_icon()
        self.add_slider_row(
            settings_layout, self.lbl_memory, self.tip_mem, self.slider_memory
        )

        self.slider_blocks = self.create_slider(
            2, 50, int(self.settings.value("blocks", 15)), self.update_labels
        )
        self.lbl_blocks, self.tip_blocks = QLabel(), self.create_tooltip_icon()
        self.add_slider_row(
            settings_layout, self.lbl_blocks, self.tip_blocks, self.slider_blocks
        )

        self.slider_padding = self.create_slider(
            0, 50, int(self.settings.value("padding", 10)), self.update_labels
        )
        self.lbl_padding, self.tip_pad = QLabel(), self.create_tooltip_icon()
        self.add_slider_row(
            settings_layout, self.lbl_padding, self.tip_pad, self.slider_padding
        )

        self.grp_settings.setLayout(settings_layout)
        self.control_layout.addWidget(self.grp_settings)

        self.control_layout.addStretch()

        self.lbl_status = QLabel()
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #4da6ff; font-weight: bold;")
        self.control_layout.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        self.control_layout.addWidget(self.progress_bar)

        # Action Buttons
        buttons_layout = QHBoxLayout()
        self.btn_start = QPushButton()
        self.btn_pause = QPushButton() # الزرار الجديد
        self.btn_stop = QPushButton()
        
        self.btn_start.setMinimumHeight(45)
        self.btn_pause.setMinimumHeight(45)
        self.btn_stop.setMinimumHeight(45)
        
        self.btn_start.setStyleSheet("background-color: #28a745; color: white; font-size: 15px; font-weight: bold; border-radius: 6px;")
        self.btn_pause.setStyleSheet("background-color: #ffc107; color: black; font-size: 15px; font-weight: bold; border-radius: 6px;")
        self.btn_stop.setStyleSheet("background-color: #dc3545; color: white; font-size: 15px; font-weight: bold; border-radius: 6px;")
        
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_pause.clicked.connect(self.toggle_pause) # ربط الزرار بالدالة
        self.btn_stop.clicked.connect(self.stop_processing)
        
        self.btn_pause.setEnabled(False) # مقفول في البداية
        self.btn_stop.setEnabled(False)

        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_pause)
        buttons_layout.addWidget(self.btn_stop)
        self.control_layout.addLayout(buttons_layout)
        
        # Video Preview Area
        video_layout = QVBoxLayout()
        self.lbl_preview = VideoDisplayLabel()
        self.lbl_preview.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.lbl_preview.setMinimumSize(640, 480)
        video_layout.addWidget(self.lbl_preview, stretch=1)

        player_layout = QHBoxLayout()
        self.check_compare = QCheckBox()
        self.check_compare.setStyleSheet("color: #4da6ff; font-weight: bold;")
        player_layout.addWidget(self.check_compare)

        self.slider_timeline = QSlider(Qt.Orientation.Horizontal)
        self.slider_timeline.setEnabled(False)
        self.slider_timeline.setTracking(False)
        self.slider_timeline.valueChanged.connect(self.scrub_video)
        self.slider_timeline.sliderMoved.connect(self.update_time_label_only)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet(
            "color: #aaa; font-weight: bold; font-family: monospace;"
        )

        player_layout.addWidget(self.slider_timeline)
        player_layout.addWidget(self.lbl_time)
        video_layout.addLayout(player_layout)

        self.main_layout.addWidget(control_panel, stretch=3)
        self.main_layout.addLayout(video_layout, stretch=7)
        self.setLayout(self.main_layout)

    def add_slider_row(self, parent_layout, lbl, tip, slider):
        h = QHBoxLayout()
        h.addWidget(lbl)
        h.addStretch()
        h.addWidget(tip)
        parent_layout.addLayout(h)
        parent_layout.addWidget(slider)

    def change_language(self, lang_name):
        if lang_name:
            self.current_lang_name = lang_name
            self.settings.setValue("language", self.current_lang_name)
            self.apply_language()

    def apply_language(self):
        is_rtl = self.available_langs.get(self.current_lang_name, {}).get("rtl", False)
        self.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft if is_rtl else Qt.LayoutDirection.LeftToRight
        )

        self.setWindowTitle(self.tr("Satrify"))
        if not self.input_file:
            self.lbl_preview.setText(self.tr("Select a video to start"))

        self.btn_input.setText(self.tr("📁 Open Video"))
        self.btn_output.setText(self.tr("💾 Save As..."))
        if not self.input_file:
            self.lbl_input.setText(self.tr("Not selected"))
        if not self.output_file:
            self.lbl_output.setText(self.tr("Not selected"))

        self.grp_mode.setTitle(self.tr("Mode"))
        self.radio_auto.setText(self.tr("🤖 Auto (Faces)"))
        self.radio_manual.setText(self.tr("🎯 Manual (Draw Box)"))
        self.radio_hybrid.setText(self.tr("🤝 Hybrid (Smart Track)"))

        self.grp_shape.setTitle(self.tr("Shape"))
        self.radio_square.setText(self.tr("Square"))
        self.radio_circle.setText(self.tr("Circle / Ellipse"))

        self.grp_settings.setTitle(self.tr("Settings"))
        self.lbl_ai.setText(self.tr("AI Model:"))
        self.btn_start.setText(self.tr("🚀 Start"))
        self.btn_pause.setText(self.tr("⏸ Pause"))
        self.btn_stop.setText(self.tr("⏹️ Stop"))
        self.lbl_status.setText(self.tr("Ready"))
        self.check_compare.setText(self.tr("Compare (Before / After)"))

        self.tip_ai.setToolTip(self.tr("tip_ai"))
        self.tip_conf.setToolTip(self.tr("tip_conf"))
        self.tip_mem.setToolTip(self.tr("tip_mem"))
        self.tip_blocks.setToolTip(self.tr("tip_blocks"))
        self.tip_pad.setToolTip(self.tr("tip_pad"))

        self.update_labels()

    def toggle_pause(self):
        # الدالة الجديدة المسؤولة عن ضغطة زرار الإيقاف المؤقت
        if hasattr(self, "processor") and self.processor.isRunning():
            is_paused = self.processor.toggle_pause()
            if is_paused:
                self.btn_pause.setText(self.tr("▶ Resume"))
            else:
                self.btn_pause.setText(self.tr("⏸ Pause"))
                
    def toggle_drawing_mode(self):
        is_manual_or_hybrid = (
            self.radio_manual.isChecked() or self.radio_hybrid.isChecked()
        )
        self.lbl_preview.drawing_enabled = is_manual_or_hybrid
        self.lbl_preview.roi_rect = None
        self.lbl_preview.update()
        self.lbl_preview.setCursor(
            Qt.CursorShape.CrossCursor
            if is_manual_or_hybrid
            else Qt.CursorShape.ArrowCursor
        )

    def create_slider(self, min_val, max_val, default_val, connect_func):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(connect_func)
        return slider

    def update_labels(self):
        conf, mem, blocks, pad = (
            self.slider_conf.value(),
            self.slider_memory.value(),
            self.slider_blocks.value(),
            self.slider_padding.value(),
        )
        self.lbl_conf.setText(f"{self.tr('Detection Confidence:')} {conf / 100.0:.2f}")
        self.lbl_memory.setText(f"{self.tr('Memory (Frames):')} {mem}")
        self.lbl_blocks.setText(f"{self.tr('Pixelation Strength (%):')} {blocks}")
        self.lbl_padding.setText(f"{self.tr('Coverage Padding (%):')} {pad}")

        self.settings.setValue("conf", conf)
        self.settings.setValue("memory", mem)
        self.settings.setValue("blocks", blocks)
        self.settings.setValue("padding", pad)

    def format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def select_input(self):
        file, _ = QFileDialog.getOpenFileName(
            None, self.tr("📁 Open Video"), "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file:
            self.input_file = file
            self.lbl_input.setText(os.path.basename(file))
            try:
                filename, _ = os.path.splitext(os.path.basename(file))
                self.output_file = get_standard_output_path(f"{filename}_Satrified.mp4")
                self.lbl_output.setText(
                    f".../Satrify_Output/{os.path.basename(self.output_file)}"
                )
            except Exception as e:
                logger.error(f"Error setting default output: {e}")

            if self.preview_cap is not None:
                self.preview_cap.release()
            self.preview_cap = cv2.VideoCapture(self.input_file)
            self.lbl_preview.video_orig_size = (
                int(self.preview_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.preview_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            self.video_fps = self.preview_cap.get(cv2.CAP_PROP_FPS) or 30.0

            self.slider_timeline.setEnabled(True)
            self.slider_timeline.setMaximum(
                int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            )
            self.slider_timeline.setValue(0)
            self.scrub_video(0)

    def update_time_label_only(self, frame_idx):
        if self.video_fps > 0:
            self.lbl_time.setText(
                f"{self.format_time(frame_idx / max(1, self.video_fps))} / {self.format_time(self.slider_timeline.maximum() / max(1, self.video_fps))}"
            )

    def scrub_video(self, frame_idx):
        if self.preview_cap and self.preview_cap.isOpened():
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = self.preview_cap.read()
            if success:
                self.update_frame(frame)
                self.lbl_preview.roi_rect = None
                self.lbl_preview.update()
                self.update_time_label_only(frame_idx)

    def select_output(self):
        file, _ = QFileDialog.getSaveFileName(
            None, self.tr("💾 Save As..."), self.output_file, "Video Files (*.mp4)"
        )
        if file:
            logger.info(f"Selected output file: {file}")
            self.output_file = file
            self.lbl_output.setText(os.path.basename(file))

    def update_frame(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.lbl_preview.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                max(self.lbl_preview.width(), 1),
                max(self.lbl_preview.height(), 1),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def update_status_from_processor(self, text):
        self.lbl_status.setText(self.tr(text))

    def update_timeline_ui(self, frame_idx):
        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setValue(frame_idx)
        self.slider_timeline.blockSignals(False)
        self.update_time_label_only(frame_idx)

    def start_processing(self):
        if not self.input_file or not self.output_file:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please select input video and output location first!"),
            )
            return

        mode = (
            "auto"
            if self.radio_auto.isChecked()
            else "manual" if self.radio_manual.isChecked() else "hybrid"
        )
        manual_bbox = self.lbl_preview.get_video_roi() if mode != "auto" else None

        if mode != "auto" and not manual_bbox:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please draw a box around the object on the screen first."),
            )
            return

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True) # تشغيل زرار الإيقاف المؤقت
        self.btn_pause.setText(self.tr("⏸ Pause")) # إعادة ضبط النص
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.slider_timeline.setEnabled(False)
        self.lbl_preview.drawing_enabled = False
        if self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None

        self.processor = VideoProcessor(
            self.input_file,
            self.output_file,
            self.slider_conf.value() / 100.0,
            self.slider_memory.value(),
            self.slider_blocks.value(),
            self.slider_padding.value(),
            mode,
            manual_bbox,
            self.slider_timeline.value(),
            "circle" if self.radio_circle.isChecked() else "square",
            (
                "caffe"
                if self.combo_ai.currentIndex() == 0
                else "tf" if self.combo_ai.currentIndex() == 1 else "haar"
            ),
        )
        self.processor.show_compare = self.check_compare.isChecked()
        self.processor.progress_update.connect(self.progress_bar.setValue)
        self.processor.status_update.connect(self.update_status_from_processor)
        self.processor.frame_update.connect(self.update_frame)
        self.processor.frame_idx_update.connect(self.update_timeline_ui)
        self.processor.finished.connect(self.on_finished)
        self.processor.start()

    def stop_processing(self):
        if hasattr(self, "processor") and self.processor.isRunning():
            self.btn_pause.setEnabled(False) # تعطيل الإيقاف المؤقت
            self.btn_stop.setEnabled(False)
            self.lbl_status.setText(self.tr("Stopping processing safely..."))
            self.processor.stop()

    def on_finished(self, success, message):
        logger.info(f"Processor finished. Success: {success}, Message: {message}")
        self.btn_pause.setEnabled(False) # تعطيل الإيقاف المؤقت
        self.btn_pause.setText(self.tr("⏸ Pause"))
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText(self.tr(message))

        if self.input_file:
            self.preview_cap = cv2.VideoCapture(self.input_file)
            self.slider_timeline.setEnabled(True)
            self.toggle_drawing_mode()

        if success:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle(self.tr("Success"))
            msg_box.setText(self.tr(message))
            btn_play = msg_box.addButton(
                self.tr("Play Video"), QMessageBox.ButtonRole.ActionRole
            )
            msg_box.addButton(self.tr("Close"), QMessageBox.ButtonRole.RejectRole)
            msg_box.exec()

            if msg_box.clickedButton() == btn_play:
                try:
                    filepath = os.path.abspath(self.output_file)
                    if sys.platform == "win32":
                        os.startfile(filepath)
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", filepath])
                    else:
                        subprocess.Popen(["xdg-open", filepath])
                except Exception as e:
                    logger.error(f"Failed to open video: {e}")
        else:
            QMessageBox.warning(self, self.tr("Warning"), self.tr(message))


def setup_dark_theme(app):
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_color, disabled_color = QColor(45, 45, 45), QColor(127, 127, 127)

    dark_palette.setColor(QPalette.ColorRole.Window, dark_color)
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, dark_color)
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_color
    )
    dark_palette.setColor(QPalette.ColorRole.Button, dark_color)
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color
    )
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)

    app.setStyleSheet(
        """
        QGroupBox { border: 1px solid #555; border-radius: 6px; margin-top: 15px; padding-top: 15px; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; color: #4da6ff; font-weight: bold; }
        QLabel { font-weight: normal; }
        QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; padding: 5px; font-size: 13px; border-radius: 4px; font-weight: normal; }
        QPushButton { border: 1px solid #555; border-radius: 4px; padding: 5px; background-color: #353535; font-weight: bold; }
        QPushButton:hover { background-color: #454545; }
        QPushButton:pressed { background-color: #252525; }
        QRadioButton { font-weight: normal; color: #fff; padding: 2px; }
        QRadioButton::indicator { width: 14px; height: 14px; }
        QComboBox { background-color: #353535; color: white; border: 1px solid #555; padding: 5px; border-radius: 4px; font-weight: normal; }
        QComboBox QAbstractItemView { background-color: #353535; color: white; selection-background-color: #2a82da; }
    """
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "satrify.png")
    app.setWindowIcon(QIcon(icon_path))
    setup_dark_theme(app)
    ex = FaceBlurApp()
    ex.show()
    sys.exit(app.exec())
