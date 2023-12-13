
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("bbpsyms3.csv")
data = np.array(data)

X = data[:, :-1]
y = data[:, -1]
# Do not convert y to int
#y = y.astype('int')
X = X.astype('int')
#print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Create a random forests classifier
rf = RandomForestClassifier()
# Fit the model on the training data
rf.fit(X_train, y_train)


inputt=[int(x) for x in "1 0 1 0 1 1 1 0 1 1 1 1 0 0 0 1 1 0 1 0 1".split(' ')]
final=[np.array(inputt)]

b = rf.predict_proba(final)

# Save the model
pickle.dump(rf,open('rfmodel2.pkl','wb'))

# Load the model
model=pickle.load(open('rfmodel2.pkl','rb'))




data_urti = pd.read_csv("urtisymptoms.csv")
data_urti = np.array(data_urti)

X_urti = data_urti[:, :-1]
y_urti = data_urti[:, -1]
# Do not convert y to int
#y = y.astype('int')
X_urti = X_urti.astype('int')
#print(X,y)
X_urti_train, X_urti_test, y_urti_train, y_urti_test = train_test_split(X_urti, y_urti, test_size=0.3, random_state=0)
# Create a random forests classifier
rf_urti = RandomForestClassifier()
# Fit the model on the training data
rf_urti.fit(X_urti_train, y_urti_train)


inputt_urti=[int(x) for x in "1 0 1 0 1 1 1 0 1 1 1 1 0 0 0 1 1".split(' ')]
final_urti=[np.array(inputt_urti)]

b_urti = rf_urti.predict_proba(final_urti)

# Save the model
pickle.dump(rf_urti,open('rfmodelurti.pkl','wb'))

# Load the model
model=pickle.load(open('rfmodelurti.pkl','rb'))





























