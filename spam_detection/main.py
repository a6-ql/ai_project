import pandas as pd
from sklearn.model_selection import train_test_split
from svm import Svm
from Logistic_Regression import logistic_regression
from D_tree import de_terr
from sklearn.preprocessing import StandardScaler
import numpy
from sklearn.feature_selection import SelectKBest, chi2
from Knn import knn

data = pd.read_csv(r'path of csv file here')
#print(type(data))
data.drop_duplicates(inplace=True)
print(data.columns)


#you have to choose comment line 18 or 19
data.drop(columns=['text'], inplace=True)
#data['text'] = data['text'].isnull().astype(int)



# Replacing All NaN Values in label_num with the Median or mean of it if exist
if data['label_num'].isnull().any():
    # Fill null values with the median of the 'label_num' column
    data['label_num'] = data['label_num'].fillna(data['label_num'].median())

if data['# sent emails '].isnull().any():
    # Fill null values with the median of the '# sent emails' column
    data['# sent emails '] = data['# sent emails '].fillna(data['# sent emails '].mean())
#print("7a7a")

# Drop other NaN Values From all Columns # sent emails
#print(type(data))
data.dropna(inplace=True)
# Converting Categorial Data to Numerical
#
data["label"] = data['label'].map({'ham': 0, 'spam': 1})
data.drop(columns=['label'], inplace=True)
print(data)
#
#
#if you choose to comment line 19
X = data.iloc[:, : 1].values

#if you choose to comment line 18
#X = data.iloc[:, : 2].values



#X.drop(columns=['label'], inplace=True)

Y = data.iloc[:,-1].values

# feature selection
#kbest = SelectKBest(chi2, k=2)
#kbest.fit_transform(X, Y)
#print(kbest.get_support())

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.3, random_state=20)

#data scaling
sc = StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
algo = logistic_regression(X_train, X_test, Y_train, Y_test)
algo.Accuracy()
algo2 = de_terr(X_train, X_test, Y_train, Y_test)
algo2.Accuracy()
algo3 = knn(X_train, X_test, Y_train, Y_test)
algo3.Accuracy()
algo4= Svm(X_train, X_test, Y_train, Y_test)
algo4.Accuracy()