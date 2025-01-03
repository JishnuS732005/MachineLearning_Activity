import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Sample email dataset
data = {
    "Email": [
        "Win a free prize now",
        "Meeting at 10am tomorrow",
        "Earn money quickly and easily",
        "Schedule for the next week",
        "Click here for a great offer",
        "Your invoice for last month",
        "Exclusive deal just for you",
        "Join our webinar on data science",
    ],
    "Label": [1, 0, 1, 0, 1, 0, 1, 0],
}
df = pd.DataFrame(data)

# Text vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Email"]).toarray()
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear SVM
linear_svm = SVC(kernel="linear", random_state=42)
linear_svm.fit(X_train, y_train)
linear_pred = linear_svm.predict(X_test)

# Non-linear SVM (RBF kernel)
nonlinear_svm = SVC(kernel="rbf", random_state=42)
nonlinear_svm.fit(X_train, y_train)
nonlinear_pred = nonlinear_svm.predict(X_test)

# Evaluation
print("Linear SVM Classification Report:")
print(classification_report(y_test, linear_pred))
print("Linear SVM Confusion Matrix:")
print(confusion_matrix(y_test, linear_pred))

print("\nNon-linear SVM Classification Report:")
print(classification_report(y_test, nonlinear_pred))
print("Non-linear SVM Confusion Matrix:")
print(confusion_matrix(y_test, nonlinear_pred))
