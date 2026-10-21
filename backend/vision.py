"""EfficientNetB0 inference with Grad-CAM. TensorFlow is only imported when a trained model exists."""
import os, json
import numpy as np
from PIL import Image
MODEL_PATH, CLASSES_PATH = os.getenv("MODEL_PATH", "/data/model.keras"), os.getenv("CLASSES_PATH", "/data/classes.json")
IMG, REVIEW, CAM_THRESHOLD = 224, float(os.getenv("REVIEW_CONF", "0.6")), 0.5
_M = {}

def available(): return os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)

def _load():
    if not _M:
        import tensorflow as tf
        m = tf.keras.models.load_model(MODEL_PATH)
        _M.update(tf=tf, classes=json.load(open(CLASSES_PATH)), gm=tf.keras.Model(m.inputs, [m.get_layer("top_activation").output, m.output]))
    return _M

def severity(cls, coverage):  # assumption: the share of the photo the model relied on stands in for how widespread the problem is
    return "low" if cls == "clean" or coverage < 25 else "medium" if coverage < 55 else "high"

def jet(v):
    return (np.stack([np.clip(1.5 - np.abs(4 * v - k), 0, 1) for k in (3, 2, 1)], -1) * 255).astype("uint8")

def overlay(img, cam, out):  # blend the heat map onto the original photo, stronger where the evidence is
    c = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize(img.size, Image.BILINEAR), dtype="float32") / 255
    a = (0.55 * c)[..., None]; mix = np.asarray(img, dtype="float32") * (1 - a) + jet(c).astype("float32") * a
    Image.fromarray(mix.astype("uint8")).save(out, quality=85)

def predict(path, cam_path=None):
    M = _load(); tf = M["tf"]; img = Image.open(path).convert("RGB")
    x = tf.convert_to_tensor(np.asarray(img.resize((IMG, IMG)), dtype="float32")[None])  # raw 0-255, EfficientNet rescales inside
    with tf.GradientTape() as tape:
        tape.watch(x); conv, probs = M["gm"](x, training=False); i = int(tf.argmax(probs[0])); score = probs[:, i]
    g = tape.gradient(score, conv); w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv * w, axis=-1))[0].numpy(); cam = cam / cam.max() if cam.max() > 0 else cam
    cls, p = M["classes"][i], probs[0].numpy(); cov = float((cam > CAM_THRESHOLD).mean() * 100)
    cam_path = cam_path or os.path.splitext(path)[0] + "_cam.jpg"; overlay(img, cam, cam_path)
    top = np.argsort(p)[::-1][:3]
    return dict(issue_class=cls, severity=severity(cls, cov), confidence=round(float(p[i]), 2), coverage_pct=round(cov, 1), needs_review=float(p[i]) < REVIEW,
                source="cnn", severity_source="grad-cam coverage", gradcam_path=cam_path, top3=[[M["classes"][k], round(float(p[k]), 2)] for k in top])
