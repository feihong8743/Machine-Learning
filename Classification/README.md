this folder is the classification models and training set

KNNClassifier_iris.ipynb

Dataset:  Sklearn.Dataset.iris
Algorithm: sklearn.neighbors.KNeighborsClassifier
Feature Scaling: sklearn.preprocessing.StandardScaler
Category Encoding: N/A
Hyperparameter: n_neighbors[1, 3, 5, 7]
Model Tuning: sklearn.model_selection.GridSearchCV
Accuracy_score: 0.94

KNNClassifier_titanic.ipynb
Dataset: tatanic
Algorithm: sklearn.neighbors.KNeighborsClassifier
Feature Scaling: sklearn.preprocessing.StandardScaler
Category Encoding: sklearn.preprocessing.OneHotEncoder
Hyperparameter: n_neighbors[1, 3, 5, 7]; algorithm':['auto', 'ball_tree', 'kd_tree']; metric' : ['minkowski','euclidean','manhattan']
Model Tuning: sklearn.model_selection.GridSearchCV
Accuracy_score: 0.81

LogisticRegression.ipynb
Dataset: tatanic
Algorithm: sklearn.linear_model.LogisticRegression
Feature Scaling: sklearn.preprocessing.StandardScaler
Category Encoding: sklearn.preprocessing.OneHotEncoder
Hyperparameter: param_grid = { 
    'penalty': ['l1', 'l2'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
}
Model Tuning: sklearn.model_selection.GridSearchCV
Accuracy_score: 0.80

Optuna Turning.ipynb
Dataset:  Sklearn.Dataset.iris
Algorithm: sklearn.neighbors.KNeighborsClassifier;
klearn.ensemble.RandomForestClassifier;
xgboost.XGBClassifier
Feature Scaling: N/A
Category Encoding: N/A
Hyperparameter: 
}
Model Tuning: Optuna
Accuracy_score: 0.97;0.95;0.94;

diabetes-classfication.ipynb
Dataset:  diabetes_data.csv
Algorithm: 
klearn.ensemble.RandomForestClassifier;
xgboost.XGBClassifier
Feature Scaling: Pipeline(steps =[('imputer_numeric', SimpleImputer(missing_values=np.nan, strategy='mean')),('scaler', StandardScaler())])
Category Encoding: Pipeline(steps=[('imputer_category', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])
Hyperparameter: 
params = {
        'randomforestclassifier__n_estimators': trail.suggest_int('randomforestclassifier__n_estimators', 10, 20, log=True),
        'randomforestclassifier__max_depth': trail.suggest_int("randomforestclassifier__max_depth", 3, 10, log=True) ,
      'randomforestclassifier__max_features': trail.suggest_categorical('randomforestclassifier__max_features', ['auto', 'sqrt']),
        'randomforestclassifier__min_samples_split': trail.suggest_int("randomforestclassifier__min_samples_split", 2, 10, log=True) ,
        'randomforestclassifier__min_samples_leaf': trail.suggest_int("randomforestclassifier__min_samples_leaf", 1, 4, log=True),
        'randomforestclassifier__bootstrap': trail.suggest_categorical('randomforestclassifier__bootstrap', [True, False])    }
 params = {
        'xgbclassifier__n_estimators': trail.suggest_int('xgbclassifier__n_estimators', 10, 20, log=True),
        'xgbclassifier__max_depth': trail.suggest_int("xgbclassifier__max_depth", 3, 20, log=True) ,
        'xgbclassifier__eta': trail.suggest_float('xgbclassifier__eta', 0.1, 0.3, log=True),
        'xgbclassifier__subsample': trail.suggest_float("xgbclassifier__subsample", 0.4, 0.8, log=True) ,
        'xgbclassifier__colsample_bytree': trail.suggest_float("xgbclassifier__colsample_bytree", 0.4, 0.8, log=True),
    }
Model Tuning: Optuna
f1_score: 0.95

body-performance-multiclass.ipynb
Dataset:  bodyPerformance.csv
Algorithm: 
klearn.ensemble.RandomForestClassifier;
xgboost.XGBClassifier
lightgbm.LGBMClassifier
Feature Scaling: Pipeline(steps =[('imputer_numeric', SimpleImputer(missing_values=np.nan, strategy='mean')),('scaler', StandardScaler())])
Category Encoding: Pipeline(steps=[('imputer_category', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])
Hyperparameter: 
params = {
       'randomforestclassifier__n_estimators': trail.suggest_int('randomforestclassifier__n_estimators', 10, 20, log=True),
       'randomforestclassifier__max_depth': trail.suggest_int("randomforestclassifier__max_depth", 3, 10, log=True) ,
    'randomforestclassifier__max_features': trail.suggest_categorical('randomforestclassifier__max_features', ['auto', 'sqrt']),
       'randomforestclassifier__min_samples_split': trail.suggest_int("randomforestclassifier__min_samples_split", 2, 10, log=True) ,
       'randomforestclassifier__min_samples_leaf': trail.suggest_int("randomforestclassifier__min_samples_leaf", 1, 4, log=True),
      'randomforestclassifier__bootstrap': trail.suggest_categorical('randomforestclassifier__bootstrap', [True, False])
   }
    
   params = {
        'xgbclassifier__n_estimators': trail.suggest_int('xgbclassifier__n_estimators', 10, 20, log=True),
       'xgbclassifier__max_depth': trail.suggest_int("xgbclassifier__max_depth", 3, 20, log=True) ,
       'xgbclassifier__eta': trail.suggest_float('xgbclassifier__eta', 0.1, 0.3, log=True),
       'xgbclassifier__subsample': trail.suggest_float("xgbclassifier__subsample", 0.4, 0.8, log=True) ,
       'xgbclassifier__colsample_bytree': trail.suggest_float("xgbclassifier__colsample_bytree", 0.4, 0.8, log=True),
     }

    params = {
        'lgbmclassifier__learning_rate': trail.suggest_float('lgbmclassifier__learning_rate', 0.1, 1.0, log=True),
        'lgbmclassifier__boosting_type"': trail.suggest_categorical("lgbmclassifier__boosting_type", ['gbdt', 'dart', 'goss']) ,
        'lgbmclassifier__sub_feature': trail.suggest_float('lgbmclassifier__sub_feature', 0.1, 1.0, log=True),
        'lgbmclassifier__num_leaves': trail.suggest_int("lgbmclassifier__num_leaves", 10, 20, log=True)
    }
Model Tuning: Optuna
accuracy: 0.73