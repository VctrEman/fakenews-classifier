import pandas as pd
import numpy as np
import os
from sklearn.svm import LinearSVC, SVC  # Faster for text than standard SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# Instead of -1, leave room for VS Code and Windows
total_cores = os.cpu_count()
safe_cores = max(1, total_cores - 2) 

print(f"Total CPU cores: {total_cores}")
print(f"Safe CPU cores: {safe_cores}")

# 1. SETUP NAMES AND RESULTS LIST
# Assuming names are like ['simple', 'lemma', 'stemmed', 'raw_regex']
preprocessing_names = os.listdir('./data')  # This will list all dataset types in the data folder
print(f"Found preprocessing types: {preprocessing_names}")
results_data = []

tfidf_norms = ['l1', 'l2']  # L1 (Manhattan) vs L2 (Euclidean)
models = [
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    ("SVM rbf", SVC(kernel='rbf', max_iter=2000, random_state=42)),
    
]

def run_comparison():
    for name in preprocessing_names:
            print(f"\n🚀 Dataset: {name}")
            
            # Load Data
            train_path = f"./data/{name}/train.csv"
            test_path = f"./data/{name}/test.csv"
            
            if not os.path.exists(train_path): continue
            
            train_df = pd.read_csv(train_path).dropna(subset=['text', 'label'])
            test_df = pd.read_csv(test_path).dropna(subset=['text', 'label'])
            
            X_train, y_train = train_df['text'], train_df["label"]
            X_test, y_test = test_df['text'], test_df["label"]

            for norm in tfidf_norms:
                # Fixed Models (Default Regularization C=1.0)
                models_to_test = [
                    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
                    ("SVM", LinearSVC(dual='auto', max_iter=2000, random_state=42))
                ]

                for model_name, classifier in models_to_test:
                    print(f"   Testing: {model_name} | TF-IDF Norm: {norm}")
                    
                    pipeline = Pipeline([
                        ("tfidf", TfidfVectorizer(
                            max_features=50000, 
                            ngram_range=(1,2),
                            norm=norm, # Testing L1 vs L2
                            dtype=np.float32 # Keeps WSL RAM low
                        )),
                        ("clf", classifier)
                    ])

                    # Train and Predict
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    f1 = f1_score(y_test, y_pred)
                    
                    # Store results
                    results_data.append({
                        "dataset": name,
                        "model": model_name,
                        "tfidf_norm": norm,
                        "f1_score": f1
                    })
                    
                    # Save progress in case of WSL timeout
                    experiment_count = os.listdir('./results')
    pd.DataFrame(results_data).to_csv(f"./results/comparison_results_{name}_{len(experiment_count)+1}.csv", index=False)

    print("\n📊 Done! Results saved to ./results/norm_comparison_results.csv")
    return pd.DataFrame(results_data)

if __name__ == "__main__":
    os.makedirs('./results', exist_ok=True)
    final_df = run_comparison()
    print(final_df)