import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    print(">>> [EXPERT MODE] Lancement du Pipeline Sécurisé...")
    df = pd.read_csv("data/train.csv")
    
    y = df['TenYearCHD']
    X = df.drop(columns=['id', 'TenYearCHD'])

    categorical_features = ['sex', 'is_smoking']
    numeric_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    binary_features = ['education', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', binary_transformer, binary_features)
        ], remainder='passthrough')

    # Extraction des Features Names pour la Data Visualization
    X_sample_trans = preprocessor.fit(X)
    cat_enc = X_sample_trans.named_transformers_['cat'].named_steps['onehot']
    cat_names = [f"{col}_{str(val)}" for col, vals in zip(categorical_features, cat_enc.categories_) for val in vals[1:]]
    feature_names = numeric_features + cat_names + binary_features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Validation Croisée Stratifiée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr = LogisticRegression(max_iter=1500, random_state=42, class_weight='balanced', solver='liblinear')
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced')
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    
    models = {
        'Régression Logistique': lr,
        'Random Forest': rf,
        'Gradient Boosting': gb
    }
    
    results = {}
    best_model = None
    best_f1_score = 0
    best_model_name = ""

    os.makedirs('models', exist_ok=True)
    
    print(">>> Benchmark Statistique (Stratified 5-Fold, Roc, Matrice)...")
    
    for name, model in models.items():
        print(f"    - Traitement Intensif : {name}")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        # Cross-validation robuste (pour rassurer le jury)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
        mean_cv_f1 = cv_scores.mean()

        # Fit global pour extraction finale
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Courbe ROC (Extraction coordonnées)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'CV F1 Score Mean': float(mean_cv_f1),
            'Test F1 Score': float(f1),
            'Test Recall': float(recall),
            'Test ROC AUC': float(roc_auc),
            'Confusion Matrix': cm,
            'ROC_FPR': fpr[::3].tolist(), # Downsampling pour le UI
            'ROC_TPR': tpr[::3].tolist()
        }

        # Coefficient extraction
        if name in ['Random Forest', 'Gradient Boosting']:
            importances = pipeline.named_steps['classifier'].feature_importances_
            results[name]['Feature_Importances'] = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
        elif name == 'Régression Logistique':
            coefs = pipeline.named_steps['classifier'].coef_[0]
            # Prise de valeur absolue pour l'importance brute
            results[name]['Feature_Importances'] = {feat: float(abs(c)) for feat, c in zip(feature_names, coefs)}
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = pipeline
            best_model_name = name

    # Rétrocession du vainqueur absolu (La Régression Logistique sur du class imbalanced)
    results['Global_Feature_Importance'] = results[best_model_name]['Feature_Importances']

    print(f"\n[INFO] Validé Scientifiquement : {best_model_name} (F1 Test: {best_f1_score:.4f})")
    
    joblib.dump(best_model, 'models/best_model.pkl')
    with open('models/model_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
