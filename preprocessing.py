import os
from packaging import metadata
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Recommended for progress tracking

from utils import SpacyPreprocessor, clean_text


# Instead of -1, leave room for VS Code and Windows
total_cores = os.cpu_count()
safe_cores = max(1, total_cores - 2) 

print(f"Total CPU cores: {total_cores}")
print(f"Safe CPU cores: {safe_cores}")

def load_fakebr(path):
    texts = []
    labels = []
    metadata = []

    meta_columns = [
    "author"
    ,"link"
    ,"category"
    ,"date of publication"
    ,"number of tokens"
    ,"number of words without punctuation"
    ,"number of types"
    ,"number of links inside the news"
    ,"number of words in upper case"
    ,"number of verbs"
    ,"number of subjuntive and imperative verbs"
    ,"number of nouns"
    ,"number of adjectives"
    ,"number of adverbs"
    ,"number of modal verbs (mainly auxiliary verbs)"
    ,"number of singular first and second personal pronouns"
    ,"number of plural first personal pronouns"
    ,"number of pronouns"
    ,"pausality"
    ,"number of characters"
    ,"average sentence length"
    ,"average word length"
    ,"percentage of news with speeling errors"
    ,"emotiveness"
    ,"diversity"]

    for label_type in ['fake', 'true']:
        folder_path = os.path.join(path, label_type)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(0 if label_type == 'fake' else 1)

    for label_type in ['fake', 'true']:
        folder_path = os.path.join(path, label_type +'-meta-information')

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines()]
                #print(lines)  
                data = dict(zip(meta_columns, lines))

                metadata.append(data)

    df = pd.DataFrame({
        'text': texts,
        'label': labels,
    })
    df_meta = pd.DataFrame(metadata)
    df = pd.concat([df, df_meta], axis=1)

    return df

# Caminho para os textos completos
data_path = "Fake.br-Corpus/full_texts"

df = load_fakebr(data_path)

print(df.shape)

from sklearn.model_selection import train_test_split



print(len(df),  len(df))


# --------------------------------------------------
# 5. DATASET GENERATION EXECUTION
# --------------------------------------------------


# Initialize Preprocessor
preprocessor_full = SpacyPreprocessor(max_tokens=50000, batch_size=1024, n_process=safe_cores)


lemmantized_df = df.copy()
clean_df = df.copy()
lemmantized_200_df = None

# --- GENERATE DATASET 1: LEMMATIZED ---
print("--- Starting Lemmatization (Deep Clean) ---")
lemmantized_df["text"] = preprocessor_full.transform(lemmantized_df["text"])
lemmantized_200_df = lemmantized_df.copy()

max_tokens = 200
lemmantized_200_df["text"] = (
    lemmantized_200_df["text"]
    .str.split()
    .str[:max_tokens]
    .str.join(" ")
)

# --- GENERATE DATASET 2: NO LEMMATIZATION ---
tqdm.pandas() # This "injects" progress_apply into pandas

print("\n--- Starting Simple Regex Cleaning ---")
# Direct apply is faster than spinning up a pipe for simple regex
clean_df["text"] = clean_df["text"].progress_apply(clean_text) if 'tqdm' in globals() else clean_df["text"].apply(clean_text)

# --------------------------------------------------
# 6. SAVE OUTPUTS
# --------------------------------------------------

dfs = {
    "lemmantized": lemmantized_df[["text", "label"]],
    'lemmantized_200': lemmantized_200_df[["text","label"]],
    "simple_cleaned": clean_df[["text", "label"]],
    "original": df
}

for name, df_i in dfs.items():
    path = f"./data/{name}"
    os.makedirs(path, exist_ok=True)
    train, test = train_test_split(df_i,  test_size=0.2, random_state=42, stratify=df_i['label'])
    pd.DataFrame(train).to_csv(f"{path}/train.csv", index=False)
    test.to_csv(f"{path}/test.csv", index=False)

print(f"\nSuccess! {len(dfs)} datasets generated:")