from sklearn import metrics
from sklearn import svm



class Svm:


    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.clf = svm.SVC(kernel='linear')  # Linear Kernel
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.clf.fit(self.X_train, self.Y_train)

    def Accuracy(self):
        Prediction = self.clf.predict(self.X_test)
        print('SVM  Accuracy is: ', metrics.accuracy_score(Prediction, self.Y_test))
        print('SVM mean squared error is : ', metrics.mean_squared_error(Prediction, self.Y_test))
        print('SVM confusion matrix is : ', metrics.confusion_matrix(Prediction, self.Y_test))
        print('SVM classification report is : ')
        print( metrics.classification_report(Prediction, self.Y_test))
        print("--------------------------------------------------")

    def pre(self, num):

        return self.clf.predict(num)