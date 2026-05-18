import pandas as pd
import dagshub
import mlflow
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# Load dan splitting data
def load_splitting_data(path):
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
def load_model(feature_train, feature_test, label_train, label_test):
    dagshub.init(repo_owner='RizkiYanuar-Tech', repo_name='', mlflow=True)

    mlflow.set_experiment("Base Model RandomForest")
    
    with mlflow.start_run(run_name='')
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        # Train
        rf.fit(feature_train, label_train)
        # Test
        y_pred = rf.predict(feature_test)
        
        accuracy = accuracy_score(label_test, y_pred)
        precision = precision_score(label_test, y_pred)
        recall = recall_score(label_test, y_pred)
        f1score = f1_score(label_test, y_pred)

        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('Precision', precision)
        mlflow.log_metric('Recall', recall)
        mlflow.log_metric('f1_score', f1score)

        mlflow.sklearn.log_model(rf, 'Random_Forest_Baseline')

        print('Model selesai dilatih dan disimpan dalam dagshub')
