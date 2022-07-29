from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

import pickle

class VoteClass():
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
    
    def fit_models(self):
        # Gradient Boosting Model
        classifier_svc = SVC(C = 1.6,
                             kernel="rbf"
        )
        classifier_svc.fit(self.X_train, self.Y_train)

        # Decision Tree
        classifier_gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=99)
        classifier_gb.fit(self.X_train, self.Y_train)

        """
        # XGBoost
        classifier_xgb = XGBClassifier(max_depth=6,learning_rate=0.1,n_estimators=1500,objective='binary:logistic',random_state=123,n_jobs=4)
        classifier_xgb.fit(self.X_train, self.Y_train)

        # SGDClassifier
        classifier_sgd = SGDClassifier(max_iter=1000,tol=1e-3,loss='log',class_weight='balanced')
        classifier_sgd = classifier_sgd.fit(self.X_train, self.Y_train)
        """
        return classifier_svc

    def fit(self):
        model = self.fit_models()

        """
        
        self.voting_class = VotingClassifier(voting="hard", estimators=model)
        self.voting_class.fit(self.X_train, self.Y_train)
        """

        self.model = model
    
    def save(self, file_name: str):
        with open(file_name, "wb") as file:
            pickle.dump(self.model, file)








