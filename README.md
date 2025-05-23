 BinaryClassificationToolkit



BinaryClassifierEval

 Descriere

Acest proiect analizează performanța mai multor modele de clasificare binară (MLP, SVM, Logistic Regression, Random Forest) aplicate unui set de date dezechilibrat. Scopul principal este evaluarea impactului pragului de decizie (threshold) asupra metrilor de performanță pentru Multi-layer Perceptron (MLP) și compararea acestuia cu alte modele clasice.



 Modele incluse

* **Multi-layer Perceptron (MLP)**
* **Support Vector Machine (SVM)**
* **Logistic Regression**
* **Random Forest**


#Dataset

* Setul de date conține 4000 de exemple, din care 211 sunt pozitive (clasa minoritară).
* Probleme de dezechilibru între clase, ceea ce influențează metricele de evaluare.
* Datele nu sunt incluse direct în repo, dar se pot înlocui cu propriul dataset cu format similar.


## Metodologie

* Se antrenează și evaluează modelele pe același set de date.
* Pentru MLP, se testează praguri de decizie diferite: 0.5, 0.6, 0.7, 0.8, 0.9.
* Se calculează și se raportează metricele: Precizie, Recall, F1-score și AUC (Area Under ROC Curve).
* Se analizează matricea de confuzie pentru pragul optim.
* Se compară performanțele MLP cu cele ale SVM, Logistic Regression și Random Forest.



## Rezultate cheie

| Prag (threshold) | Precizie | Recall | F1-score |
| ---------------- | -------- | ------ | -------- |
| 0.5              | 0.236    | 0.967  | 0.379    |
| 0.6              | 0.236    | 0.967  | 0.379    |
| 0.7              | 0.290    | 0.801  | 0.426    |
| 0.8              | 0.373    | 0.697  | 0.486    |
| 0.9              | 0.000    | 0.000  | 0.000    |

* Pragul optim pentru MLP este **0.7**, cu un F1-score de aproximativ 0.426.
* AUC pentru MLP la pragul 0.7: **0.940**.
* Alte modele au AUC-uri comparabile, dar F1-score-uri mai scăzute pentru clasa minoritară.


## Matrice de confuzie exemplu (MLP, prag=0.7)

|          | Predicted 0 | Predicted 1 |
| -------- | ----------- | ----------- |
| Actual 0 | 3374        | 415         |
| Actual 1 | 42          | 169         |



## Observații

* Pentru praguri mari (ex: 0.9), nu se prezic exemple pozitive, ceea ce duce la precizie zero — avertismentele din sklearn sunt normale și gestionate.
* Codul este modular și ușor de extins pentru alte modele sau seturi de date.
* Recomandat pentru proiecte de învățare automată cu date dezechilibrate.




