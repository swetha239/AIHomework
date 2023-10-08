# Result classification of students in schools whether they will pass or fail
#
# Problem statement : To classify the students whether they will pass or fail in the exam based on other factors.
#
# Model implemented : Decision Tree Classifier
#
#Importing the required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the dataset
data = pd.read_csv('https://raw.githubusercontent.com/Deepsphere-AI/DSAI_WorkShop/main/APR_2023_SCHOOLS/Classification/Problem%201/Datasets/PassorfailTRAIN.csv')
data

#Model building
X = data.drop(['pass'],axis=1)
y = data['pass']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a Decision Tree Classifier object
clf = DecisionTreeClassifier()

# Train the model using the training data
clf.fit(X_train, y_train)

#Testing our model with the same dataset
predictions = clf.predict(X_test)
print("Accuracy_Score =", format(metrics.accuracy_score(y_test, predictions)))

print('\n************************************************************************************\n')

#Data visualization with binary data
print('Count Plot\n')
sns.countplot(x='pass', data=data, )
plt.show()
