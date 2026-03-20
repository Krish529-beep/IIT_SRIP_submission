import pyarrow.parquet as pq
import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


# basic text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# -------------------------------
# settings (can change if needed)
# -------------------------------
data_path = "/dataset_10M.parquet" # path to dataset (change if needed)
limit = 500000        # used only part of data due to system limits
batch_size = 50000


print("starting to read dataset...")

parquet_file = pq.ParquetFile(data_path)

data_parts = []
rows_loaded = 0

# reading in chunks to avoid memory issues
for batch in parquet_file.iter_batches(batch_size=batch_size):
    temp_df = batch.to_pandas()

    temp_df = temp_df[["DATA", "TOPIC"]].dropna()
    temp_df["DATA"] = temp_df["DATA"].apply(clean_text)

    data_parts.append(temp_df)
    rows_loaded += len(temp_df)

    print("loaded rows:", rows_loaded)

    if rows_loaded >= limit:
        break

df = pd.concat(data_parts, ignore_index=True)

print("final dataset size:", len(df))


# shuffle data (important)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


print("splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    df["DATA"], df["TOPIC"], test_size=0.2, random_state=42
)


print("applying tf-idf...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("training model...")

model = SGDClassifier(
    loss="log_loss",
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


print("evaluating model...")

y_pred = model.predict(X_test_tfidf)

print("accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# saving model
print("saving model...")

os.makedirs("final_models", exist_ok=True)

joblib.dump(model, "final_models/model.pkl")
joblib.dump(vectorizer, "final_models/vectorizer.pkl")

print("done 👍")