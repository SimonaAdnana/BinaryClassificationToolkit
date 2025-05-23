import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from scikeras.wrappers import KerasClassifier  # updated wrapper

# 1. Import CSV files
files = [
    "C:/Users/simona/Desktop/proiect_inteligenta/RO_LFS_2010_Y.csv",
    "C:/Users/simona/Desktop/proiect_inteligenta/RO_LFS_2011_Y.csv",
    "C:/Users/simona/Desktop/proiect_inteligenta/RO_LFS_2012_Y.csv",
    "C:/Users/simona/Desktop/proiect_inteligenta/RO_LFS_2013_Y.csv"
]

datasets = [pd.read_csv(file) for file in files]

# 2. Data cleaning
variabile_selectate = [
    "ILOSTAT", "SEX", "AGE", "EDUCSTAT", "EDUCLEVL", "COUNTRYB",
    "STAPRO", "TEMP", "FTPT", "WISHMORE", "LOOKOJ", "NACE1D", "ISCO1D",
    "DURUN", "HWUsual", "HWActual"
]

categorice = ["SEX", "EDUCSTAT", "EDUCLEVL", "COUNTRYB", "STAPRO", "TEMP",
              "FTPT", "WISHMORE", "LOOKOJ", "NACE1D", "ISCO1D"]

numerice = ["AGE", "DURUN", "HWUsual", "HWActual"]

lista_date_curatate = []

for df in datasets:
    existente = [var for var in variabile_selectate if var in df.columns]
    df = df[existente]

    for var in categorice:
        if var in df.columns:
            df.loc[:, var] = df[var].astype(object).astype(str).fillna('unknown').astype('category')

    for var in numerice:
        if var in df.columns:
            df.loc[:, var] = df[var].fillna(df[var].mean())

    if 'AGE' in df.columns:
        df = df[(df['AGE'] >= 15) & (df['AGE'] <= 64)]

    lista_date_curatate.append(df)

date_finale = pd.concat(lista_date_curatate, ignore_index=True)

# 3. Prepare for training
date_finale = date_finale[date_finale['ILOSTAT'].isin([1, 2, 3])]
date_finale['TARGET'] = (date_finale['ILOSTAT'] == 2).astype(int)
date_finale.drop(columns=['ILOSTAT'], inplace=True)
date_finale = pd.get_dummies(date_finale, drop_first=True)

# Sample for training
date_sample = date_finale.sample(n=20000, random_state=123)
X = date_sample.drop(columns=['TARGET'])
y = date_sample['TARGET']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

# 4. Handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 5. Scaling
scaler_mlp = MinMaxScaler()
X_train_mlp = scaler_mlp.fit_transform(X_train_smote)
X_test_mlp = scaler_mlp.transform(X_test)

scaler_svm = StandardScaler()
X_train_svm = scaler_svm.fit_transform(X_train_smote)
X_test_svm = scaler_svm.transform(X_test)

scaler_rf = StandardScaler()
X_train_rf = scaler_rf.fit_transform(X_train_smote)
X_test_rf = scaler_rf.transform(X_test)

scaler_lr = StandardScaler()
X_train_lr = scaler_lr.fit_transform(X_train_smote)
X_test_lr = scaler_lr.transform(X_test)

# Evaluation function
def evaluare_model(y_true, y_pred, y_probs=None, model_name='Model'):
    print(f"Confusion Matrix - {model_name}")
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=plt.cm.Blues)
    plt.show()
    print(classification_report(y_true, y_pred))
    if y_probs is not None:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc_score = auc(fpr, tpr)
        print(f"AUC {model_name}: {auc_score:.3f}")
        return fpr, tpr, auc_score
    return None, None, None

# 6. MLP model
mlp_model = Sequential([
    Dense(32, input_dim=X_train_mlp.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
mlp_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
mlp_model.fit(X_train_mlp, y_train_smote, epochs=50, batch_size=64, verbose=0,
              validation_split=0.2, callbacks=[early_stop])

mlp_probs = mlp_model.predict(X_test_mlp).flatten()

# Test praguri diferite pentru MLP
print("Evaluare MLP pentru praguri diferite:\n")
praguri = [0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in praguri:
    mlp_pred = (mlp_probs > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, mlp_pred, average='binary')
    print(f"Prag: {threshold:.1f} | Precizie: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f}")

# Alege pragul care îți convine și evaluează complet
best_threshold = 0.7
mlp_pred = (mlp_probs > best_threshold).astype(int)
fpr_mlp, tpr_mlp, auc_mlp = evaluare_model(y_test, mlp_pred, mlp_probs, f'MLP (prag={best_threshold})')

# 7. SVM
svm_model = SVC(kernel='rbf', C=1, gamma=0.01, probability=True, class_weight='balanced')
svm_model.fit(X_train_svm, y_train_smote)
svm_pred = svm_model.predict(X_test_svm)
svm_probs = svm_model.predict_proba(X_test_svm)[:, 1]
fpr_svm, tpr_svm, auc_svm = evaluare_model(y_test, svm_pred, svm_probs, 'SVM')

# 8. Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_lr, y_train_smote)
lr_pred = lr_model.predict(X_test_lr)
lr_probs = lr_model.predict_proba(X_test_lr)[:, 1]
fpr_lr, tpr_lr, auc_lr = evaluare_model(y_test, lr_pred, lr_probs, 'Logistic Regression')

# 9. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_rf, y_train_smote)
rf_pred = rf_model.predict(X_test_rf)
rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]
fpr_rf, tpr_rf, auc_rf = evaluare_model(y_test, rf_pred, rf_probs, 'Random Forest')

# 10. ROC Comparison
plt.figure(figsize=(10, 7))
plt.plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC={auc_mlp:.3f})', color='blue')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC={auc_svm:.3f})', color='red')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})', color='green')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.3f})', color='purple')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Modele comparate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Cross-validation for MLP using SciKeras
print("Cross-validation MLP (20% din datele de antrenament)")

def create_mlp():
    model = Sequential([
        Dense(32, input_shape=(X_train_mlp.shape[1],), activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

mlp_clf = KerasClassifier(
    model=create_mlp,
    epochs=20,
    batch_size=64,
    verbose=0
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(mlp_clf, X_train_mlp, y_train_smote, cv=cv, scoring='accuracy')
print(f"MLP Cross-validation accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}")
