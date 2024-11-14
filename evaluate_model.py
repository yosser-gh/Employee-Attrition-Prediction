import pandas as pd
from data_preprocessing import encode , normalize
from train_model import train_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import joblib

#Evaluate each model and store metrics
def evaluate_models(models, X_test, y_test):
    results = []
    
    for model_name, model in models.items():
        # Predict on the test set
        predictions = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        # Only calculate AUC-ROC if model has predict_proba
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probabilities)
        else:
            auc = "N/A"
        
        # Append results for this model
        results.append({
            "Model": model_name,
            "Model_Object": model,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "AUC-ROC": auc
        })
        
    return pd.DataFrame(results)

#Calculate Composite Score based on Weighted Average
def calculate_composite_scores(results, weights=None):
    if weights is None:
        weights = {
            "Accuracy": 0.3,
            "F1 Score": 0.4,
            "AUC-ROC": 0.3
        }
    
    results['Composite Score'] = (
        results['Accuracy'] * weights['Accuracy'] +
        results['F1 Score'] * weights['F1 Score'] +
        results['AUC-ROC'] * weights['AUC-ROC']
    )
    return results

#Select and save the best model based on Composite Score
def save_best_model(results):
    best_model_row = results.loc[results['Composite Score'].idxmax()]
    best_model = best_model_row['Model_Object']
    
    # Save the best model to model/best_model.pkl
    joblib.dump(best_model, './model/best_model.pkl')
    print("model saved succesfuly")


if __name__ == '__main__':

    #Load and process the data
    df = pd.read_csv('./Data/Employee-Attrition.csv')
    df = encode(df)
    df = normalize(df)
    
    #Train the model
    log_model, svm_model, knn_model, xgb_model, rf_model, X_test, y_test = train_model(df)
    
    #Put models in a dictionary for easy access
    models = {
        "Logistic Regression": log_model,
        "SVM": svm_model,
        "KNN": knn_model,
        "XGBoost": xgb_model,
        "Random Forest": rf_model
    }
    
    #Evaluate the models
    results = evaluate_models(models, X_test, y_test)
    results = calculate_composite_scores(results)
    save_best_model(results)
