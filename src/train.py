from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def train_model(X_train, y_train, preprocessor):
    """
    Trains a RandomForestClassifier with hyperparameter tuning.
    """
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"Best hyperparameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_
