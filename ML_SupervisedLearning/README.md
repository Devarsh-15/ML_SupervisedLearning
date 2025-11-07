# ML_SupervisedLearning

### Linear Regression, Ridge & Lasso

In this session, I implemented **Linear Regression** to predict housing prices using the California Housing dataset.  
Explored **overfitting** and applied **Ridge (L2)** and **Lasso (L1)** regularization to improve generalization.  
Evaluated performance using **R² Score** and **RMSE**, visualized predicted vs actual values, and tuned hyperparameter `alpha` using GridSearchCV.  
Final models and predictions were saved for comparison with future algorithms.

### Logistic Regression (Binary Classification)

Implemented Logistic Regression using the Breast Cancer dataset to classify tumors as malignant or benign.
Covered sigmoid function, decision boundaries, probability thresholds, and regularization effects.
Evaluated model using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
Performed hyperparameter tuning for 'C' and visualized ROC curve. Model and predictions were saved for future comparison.

### K-Nearest Neighbors (KNN)

Implemented KNN classification using the Iris dataset. Covered distance metrics,
choosing optimal K, scaling importance, and visualized confusion matrix and K vs Accuracy curve.
Evaluated the model using accuracy, precision, recall, and F1-score.
Performed hyperparameter tuning using GridSearchCV and saved the trained model.

### SVM (Classification) + SVR (Regression)

Implemented Support Vector Machine (SVM) for tumor classification and Support Vector Regression (SVR) for predicting house prices.
Covered margin maximization, hyperplanes, kernels (linear, RBF, polynomial), and the effect of C and gamma on decision boundaries.
Evaluated performance using accuracy, precision, recall, F1-score, and confusion matrix.
Performed hyperparameter tuning using GridSearchCV and saved the trained model and predictions.

### Decision Trees (Classification + Regression)

Implemented DecisionTreeClassifier on the Iris dataset and DecisionTreeRegressor on
California Housing. Covered impurity measures (gini, entropy), tree splitting, node
criteria, hyperparameters such as max_depth, min_samples_split, and min_samples_leaf.
Performed evaluation using accuracy, precision, recall, F1-score, and confusion matrix,
and tuned hyperparameters using GridSearchCV. Saved model and predictions for future comparison.

### Random Forest (Classification + Regression)

Implemented RandomForestClassifier using the Iris dataset and RandomForestRegressor on
California Housing. Explained bagging, random feature sampling, variance reduction, and
overfitting control. Evaluated the model using accuracy, precision, recall, and F1-score,
and performed hyperparameter tuning for n_estimators, max_depth, and max_features using GridSearchCV.
Saved model, predictions, and all visualizations.

### AdaBoost (Boosting)

Implemented AdaBoostClassifier using Decision Stumps as weak learners. Covered boosting
concepts, weighted errors, iterative learning, and explored parameters like n_estimators
and learning_rate. Evaluated performance using accuracy, precision, recall, and F1-score,
visualized a confusion matrix, and performed hyperparameter tuning. Saved model and predictions.

### Gradient Boosting (GBDT)

Implemented GradientBoostingClassifier with deep explanations of sequential tree building,
gradient-based optimization, and parameters such as learning_rate, n_estimators, and max_depth.
Evaluated performance using accuracy, precision, recall, and F1-score, and performed
hyperparameter tuning. Also implemented GradientBoostingRegressor for housing data.
Saved models and predictions for comparison.

### XGBoost (Extreme Gradient Boosting)

Implemented XGBClassifier with detailed parameter explanations and learned second-order
optimization, regularization, shrinkage, subsampling, and column sampling. Evaluated the
model using accuracy, precision, recall, and F1-score, visualized confusion matrix, and
performed hyperparameter tuning. Also implemented XGBRegressor for regression tasks.
Saved model and predictions for future comparison.
