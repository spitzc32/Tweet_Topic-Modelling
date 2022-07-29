from sklearn.linear_model import SGDClassifier, LogisticRegression
import pickle


class ClassModel():
    def __init__(self, X_train, Y_train) -> None:
        self.X_train = X_train
        self.Y_train = Y_train

    def fit(self):
        classifcation_lr = LogisticRegression(
                C=0.75, 
                penalty='l2',
                solver = 'liblinear'
            )
        classifcation_lr.fit(self.X_train, self.Y_train)
    
        # Logistic Regression SGD
        classifcation_sgd = SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            loss='log',
            class_weight='balanced'
        )
        classifcation_sgd.fit(self.X_train, self.Y_train)

         # SGD Modified Huber
        classifcation_sgd_huber = SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            alpha=20,
            loss='modified_huber',
            class_weight='balanced'
        )
        classifcation_sgd_huber.fit(self.X_train, self.Y_train)

        return [("LogisticRegressionFreq",classifcation_lr), ("SGDFreq",classifcation_sgd), ("SGDHuberFreq",classifcation_sgd_huber)]
    
    def save(self, file_name, model):
        with open(file_name, "wb") as file:
            pickle.dump(model, file)


