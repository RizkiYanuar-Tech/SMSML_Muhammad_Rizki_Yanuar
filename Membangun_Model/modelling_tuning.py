import pandas as pd
import dagshub
import mlflow
from dotenv import load_dotenv
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

load_dotenv()

# Load dan splitting data
def load_splitting_data(path):
    """
    Membaca dataset dari file CSV, memisahkan fitur dan target (Churn), 
    serta melakukan proporsi pembagian data untuk pelatihan dan pengujian.

    Args:
        filepath (str): Jalur atau lokasi file dataset CSV (contoh: 'preprocessing/clean_data.csv').

    Returns:
        tuple: Mengembalikan 4 buah variabel data dalam bentuk tuple:
            - X_train (DataFrame): Fitur yang digunakan untuk melatih mesin.
            - X_test (DataFrame): Fitur yang digunakan untuk menguji mesin.
            - y_train (Series): Kunci jawaban (target) untuk proses pelatihan.
            - y_test (Series): Kunci jawaban (target) untuk proses pengujian.
    """
    dataset = pd.read_csv(path)

    X = dataset.drop(columns=['Churn'])
    y = dataset['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3, 
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

# Load Model
def load_train_model(feature_train, feature_test, label_train, label_test):
    """
    Melatih algoritma Random Forest Classifier dan secara otomatis mengirimkan 
    laporan performa (metrik) beserta file model akhir ke server DagsHub melalui MLflow.

    Args:
        X_train (DataFrame): Data fitur pelatihan.
        X_test (DataFrame): Data fitur pengujian.
        y_train (Series): Target/label data pelatihan.
        y_test (Series): Target/label data pengujian.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai di dalam script, melainkan 
              langsung menyimpan artefak dan metrik ke cloud (DagsHub).
    """
    dagshub.init(repo_owner='RizkiYanuar-Tech',
                 repo_name='SMSML_Muhammad_Rizki_Yanuar',
                 mlflow=True
                )

    mlflow.set_experiment("Tuning RandomForest")
    with mlflow.start_run(run_name='Tuning_RandomForest'):
        parameters = {
            'n_estimators': [50, 70, 80],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [5, 10, 20, 30],
            'min_samples_split': [3, 5, 7]
        }
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')

        random_search=RandomizedSearchCV(
            estimator=rf,
            param_distributions=parameters,
            n_iter=10, # 10 Kombinasi acak
            cv=5, # Cross Validation
            scoring='recall', # Penilaian yang diutamakan
            random_state=42,
        )

        # Train
        random_search.fit(feature_train, label_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        # Test
        y_pred = best_model.predict(feature_test)

        accuracy = accuracy_score(label_test, y_pred)
        precision = precision_score(label_test, y_pred)
        recall = recall_score(label_test, y_pred)
        f1score = f1_score(label_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('Precision', precision)
        mlflow.log_metric('Recall', recall)
        mlflow.log_metric('f1_score', f1score)

        mlflow.sklearn.log_model(best_model, 'Tuning_RandomForest')

        print('Model selesai dilatih dan disimpan dalam dagshub')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_splitting_data('clean_data.csv')
    load_train_model(X_train, X_test, y_train, y_test)
