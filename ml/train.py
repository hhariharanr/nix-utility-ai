"""Train EfficientNetB0 (ImageNet weights) in two stages: new head on a frozen base, then fine-tune the top layers.
Usage: python train.py --data dataset_split --out ../data     (writes model.keras, classes.json, report.txt, confusion_matrix.png)"""
import argparse, json, os
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
IMG = 224
ALLOWED = {"clean", "dust", "bird_droppings", "shading", "snow", "physical_damage", "electrical_damage"}

def load(path, batch, shuffle, names=None):
    return keras.utils.image_dataset_from_directory(path, image_size=(IMG, IMG), batch_size=batch, shuffle=shuffle, seed=42, class_names=names, label_mode="int")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="dataset_split"); ap.add_argument("--out", default="../data")
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--epochs-head", type=int, default=8); ap.add_argument("--epochs-ft", type=int, default=12)
    ap.add_argument("--unfreeze", type=int, default=40, help="how many top layers of the base to fine-tune"); ap.add_argument("--lr-head", type=float, default=1e-3); ap.add_argument("--lr-ft", type=float, default=1e-5)
    a = ap.parse_args(); keras.utils.set_random_seed(42); os.makedirs(a.out, exist_ok=True)
    train = load(f"{a.data}/train", a.batch, True); classes = train.class_names; n = len(classes)
    bad = set(classes) - ALLOWED
    if bad: raise SystemExit(f"Class names {sorted(bad)} are not known to the risk engine. Rename them with prepare_data.py --map.")
    val, test = load(f"{a.data}/val", a.batch, False, classes), load(f"{a.data}/test", a.batch, False, classes)
    counts = np.array([len(os.listdir(f"{a.data}/train/{c}")) for c in classes]); cw = {i: float(counts.sum() / (n * c)) for i, c in enumerate(counts)}
    print("classes:", dict(zip(classes, counts.tolist())))
    aug = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.08), layers.RandomZoom(0.1),
                            layers.RandomContrast(0.15), layers.RandomBrightness(0.15, value_range=(0, 255))])
    AUTO = tf.data.AUTOTUNE
    train_aug = train.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTO).prefetch(AUTO)
    val, test_pf = val.prefetch(AUTO), test.prefetch(AUTO)

    # EfficientNetB0 expects raw 0-255 pixels (it rescales internally), so no extra preprocessing anywhere.
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
    for l in base.layers: l.trainable = False
    x = layers.Dropout(0.3)(layers.GlobalAveragePooling2D()(base.output))
    model = keras.Model(base.input, layers.Dense(n, activation="softmax")(x))  # flat graph so Grad-CAM can reach 'top_activation'
    cb = lambda: [keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True), keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3)]
    def compile_(lr): model.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])

    compile_(a.lr_head); print("\nStage 1: training the new head on the frozen base")
    model.fit(train_aug, validation_data=val, epochs=a.epochs_head, class_weight=cw, callbacks=cb())
    for l in base.layers[-a.unfreeze:]:
        if not isinstance(l, layers.BatchNormalization): l.trainable = True  # keep BatchNorm frozen while fine-tuning
    compile_(a.lr_ft); print("\nStage 2: fine-tuning the top layers")
    model.fit(train_aug, validation_data=val, epochs=a.epochs_ft, class_weight=cw, callbacks=cb())

    model.save(f"{a.out}/model.keras"); json.dump(classes, open(f"{a.out}/classes.json", "w"))
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    y = np.concatenate([b.numpy() for _, b in test]); p = model.predict(test_pf, verbose=0).argmax(1)
    rep = classification_report(y, p, target_names=classes, digits=3); print("\nTest set\n" + rep); open(f"{a.out}/report.txt", "w").write(rep)
    cm = confusion_matrix(y, p); fig, ax = plt.subplots(figsize=(7, 6)); ax.imshow(cm, cmap="Blues"); ax.set_xticks(range(n), classes, rotation=45, ha="right"); ax.set_yticks(range(n), classes)
    for i in range(n):
        for j in range(n): ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); fig.tight_layout(); fig.savefig(f"{a.out}/confusion_matrix.png", dpi=130)
    print(f"\nSaved {a.out}/model.keras, classes.json, report.txt, confusion_matrix.png")
if __name__ == "__main__": main()
