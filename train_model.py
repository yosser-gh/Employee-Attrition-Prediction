from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def train_model(df):
    
    """**Train-Test Split**"""

    # Split the dataset into features (X) and target (y)
    X = df.drop(columns='Attrition')
    y = df['Attrition']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    """**Logistic Regression**"""

    log_model = LogisticRegression(max_iter=500)
    log_model.fit(X_train, y_train)

    
    """**Random Forest**"""

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)


    """**Support Vector Machine (SVM)**"""

    svm_model = SVC(probability=True)
    svm_model.fit(X_train, y_train)


    """**KNN classifier**"""

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)


    """**XGBoost classifier**"""

    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)


    return log_model, svm_model, knn_model,xgb_model, rf_model, X_test, y_test
