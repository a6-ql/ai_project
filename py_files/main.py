import pandas as pd
from sklearn.model_selection import train_test_split
from svm import Svm
from Logistic_Regression import logistic_regression
from D_tree import de_terr
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from Knn import knn
import tkinter as tk
from tkinter import messagebox
import joblib
data = pd.read_csv(r'C:\Users\WIN-10\Downloads\spam_ham_dataset.csv')
#print(type(data))
data.drop_duplicates(inplace=True)

#you have to choose comment line 18 or 19
data.drop(columns=['text'], inplace=True)
#data['text'] = data['text'].isnull().astype(int)



# Replacing All NaN Values in label_num with the Median or mean of it if exist
if data['label_num'].isnull().any():
    # Fill null values with the median of the 'label_num' column
    data['label_num'] = data['label_num'].fillna(data['label_num'].median()) #median 2ktr 7aga mtkrra

if data['# sent emails '].isnull().any():
    # Fill null values with the median of the '# sent emails' column
    data['# sent emails '] = data['# sent emails '].fillna(data['# sent emails '].mean())#avg 34an normal dist mtl34 outliers


# Drop other NaN Values From all Columns # sent emails
#print(type(data))
data.dropna(inplace=True)
# Converting Categorial Data to Numerical
#
data["label"] = data['label'].map({'ham': 0, 'spam': 1})#malhash lazma
data.drop(columns=['label'], inplace=True)

#
#
#if you choose to comment line 19
X = data.iloc[:, : 1].values

#if you choose to comment line 18
#X = data.iloc[:, : 2].values





Y = data.iloc[:,-1].values

# feature selection
print(data.corr())
#kbest = SelectKBest(chi2, k=1)
#kbest.fit_transform(X, Y)
#print(kbest.get_support())

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.3, random_state=20)

#data scaling
sc = StandardScaler(copy=True, with_mean=True, with_std=True)#100 1000000 1000
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

def classify_spam_ham():
    email_name = entry_email_name.get()

    #if you commented 18
    pred =[[int(entry_num_sent_emails.get()), 0]]

    #if you commented 19
    pred = [[int(entry_num_sent_emails.get())]]
    df = pd.DataFrame(pred)
    selected_algo = algo_var.get()

    # Select algo user's choice
    if selected_algo == "logistic_Regression":
        prediction = algo.pre(pred)
    elif selected_algo == "Decision Tree":
        prediction = algo2.pre(pred)
    elif selected_algo == "KNN":
        prediction = algo3.pre(pred)
    elif selected_algo == "SVM":
        prediction = algo4.pre(pred)
    else:
        messagebox.showerror("Error", "Please select an algorithm.")
        return


    # Display res
    if prediction == 1:
        messagebox.showinfo("Result", f"{email_name} is predicted as SPAM.")
    else:
        messagebox.showinfo("Result", f"{email_name} is predicted as HAM.")

# Create main window
root = tk.Tk()
root.title("Email Spam Classification")

# Create  entry fields
label_email_name = tk.Label(root, text="Email Name:")
label_email_name.grid(row=0, column=0, padx=10, pady=5)

entry_email_name = tk.Entry(root)
entry_email_name.grid(row=0, column=1, padx=10, pady=5)

label_num_sent_emails = tk.Label(root, text="Number of Sent Emails:")
label_num_sent_emails.grid(row=1, column=0, padx=10, pady=5)

entry_num_sent_emails = tk.Entry(root)
entry_num_sent_emails.grid(row=1, column=1, padx=10, pady=5)

# Create menu for algo selection
algo_var = tk.StringVar(root)
algo_var.set("logistic_Regression")  # Default algo

label_algorithm = tk.Label(root, text="Select Algorithm:")
label_algorithm.grid(row=2, column=0, padx=10, pady=5)

dropdown_algo = tk.OptionMenu(root, algo_var, "logistic_Regression", "Decision Tree", "KNN", "SVM")
dropdown_algo.grid(row=2, column=1, padx=10, pady=5)

# Create  classify button
classify_button = tk.Button(root, text="Classify", command=classify_spam_ham)
classify_button.grid(row=3, columnspan=2, pady=10)

# Start GUI main loop
root.mainloop()