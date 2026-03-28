# src/train_model.py

from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.model import build_model
from config.settings import TRAIN_PATH,LOCAL_MODEL_PATH, EPOCHS, BATCH_SIZE

def train():
    print("Loading data from:", TRAIN_PATH)

    X, y = load_data(TRAIN_PATH)

    if len(X) == 0:
        print("No data loaded. Check your dataset.")
        return

    print(" Data shape:", X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(" Building model...")
    model = build_model()

    print(" Training started...\n")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    print("\n Saving model...")
    model.save(LOCAL_MODEL_PATH, include_optimizer=False)

    print(" Training complete!")
    print(" Model saved at:", LOCAL_MODEL_PATH)


if __name__ == "__main__":
    train()