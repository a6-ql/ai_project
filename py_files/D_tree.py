from sklearn import tree
from sklearn import metrics




class de_terr:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.model = tree.DecisionTreeClassifier(random_state=1)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.model.fit(self.X_train, self.Y_train)

    def Accuracy(self):
        Prediction =self.model.predict(self.X_test)
        print('Decision Tree Accuracy is : ', metrics.accuracy_score(self.Y_test,Prediction))
        print('Decision Tree mean squared error is : ', metrics.mean_squared_error(Prediction, self.Y_test))
        print('Decision Tree confusion matrix is : ', metrics.confusion_matrix(Prediction, self.Y_test))
        print('Decision Tree classification report is : ')
        print(metrics.classification_report(Prediction, self.Y_test))
        print("--------------------------------------------------")

    def pre( self ,num):
        return self.model.predict(num)