"""Score the trained model on any folder of images arranged as <folder>/<class>/*.jpg (e.g. your own real-world photos).
Usage: python evaluate.py --data realworld --model ../data"""
import argparse, json, os
import numpy as np, tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--model", default="../data"); ap.add_argument("--review", type=float, default=0.6); a = ap.parse_args()
m = tf.keras.models.load_model(f"{a.model}/model.keras"); classes = json.load(open(f"{a.model}/classes.json")); y, p, conf = [], [], []
for c in sorted(os.listdir(a.data)):
    if c not in classes: print("skipping folder not in the model's classes:", c); continue
    for f in os.listdir(f"{a.data}/{c}"):
        try: x = np.asarray(Image.open(f"{a.data}/{c}/{f}").convert("RGB").resize((224, 224)), dtype="float32")[None]
        except Exception: continue
        pr = m.predict(x, verbose=0)[0]; y.append(classes.index(c)); p.append(int(pr.argmax())); conf.append(float(pr.max()))
y, p, conf = map(np.array, (y, p, conf)); print(f"{len(y)} images, accuracy {(y == p).mean():.3f}, would go to human review: {(conf < a.review).mean():.0%}")
labels = sorted(set(y) | set(p)); print(classification_report(y, p, labels=labels, target_names=[classes[i] for i in labels], digits=3, zero_division=0)); print(confusion_matrix(y, p, labels=labels))
