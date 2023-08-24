
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("bbp_symptoms.csv")
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
pickle.dump(rf,open('rfmodel.pkl','wb'))

# Load the model
model=pickle.load(open('rfmodel.pkl','rb'))















