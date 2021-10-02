"""

    My ML playground

"""
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# First we read the data from the csv files
data_train = pd.read_csv('titanic/train.csv')
data_test = pd.read_csv('titanic/test.csv')

# visilyze the given data
print(data_train.head())

# Note : The Survived column  is what we’re trying to predict. We call this column the (target).
# Note: The Remaining columns are called (features)

# count the number of the Survived and the deaths
print(data_train['Survived'].value_counts())  # (342 Survived) | (549 not survived)

# plot the amount of the survived and the deaths
plt.figure(figsize=(5, 5))
print(data_train['Survived'].value_counts().keys())
plt.bar(list(data_train['Survived'].value_counts().keys()), (list(data_train['Survived'].value_counts())),
        color=['r', 'g'])

# we analyze the Pclass feature
plt.figure(figsize=(5, 5))
plt.bar(list(data_train['Pclass'].value_counts().keys()), (list(data_train['Pclass'].value_counts())),
        color=['Orange', 'Blue', 'Red'])

# analyze the age
plt.figure(figsize=(5, 7))
plt.hist(data_train['Age'], color='Purple')
plt.title('Age Distribuation')
plt.xlabel('Age')
plt.show()
plt.savefig('age.jpg')

# Now after we made some analyze here and their, it's time to clean up our data
# If you take a look to the avalible columns we you may noticed that some columns are useless so they may affect
# on our model performance

# Here we make our cleaning function
def clean(data):
    # here we drop the unwanted data
    data = data.drop(['Ticket', 'Cabin', 'Name'], axis=1)
    cols = ['SibSp', 'Parch', 'Fare', 'Age']

    # Fill the Null Values with the mean value
    for col in cols:
        data[col].fillna(data[col].mean(), inplace=True)

    # fill the Embarked null values with an unknown data
    data.Embarked.fillna('U', inplace=True)
    return data


# now we call our function and start cleaning!
data_train = clean(data_train)
data_test = clean(data_test)

# now we need to change the sex feature into a numeric value like [1] for male and [0] female and
# also for the Embarked feature

# Note: we imported [from skleran import preprocessing]
le = preprocessing.LabelEncoder()
cols = ['Sex', 'Embarked'].predic
for col in cols:
    data_train[col] = le.fit_transform(data_train[col])
    data_test[col] = le.fit_transform(data_test[col])

# now our data is ready!
# it's time to build our model
# we use the following import [from sklearn.linear_model import LogisticRegression]

# we select the target column ['Survived'] to store it in [Y] and drop it from the original data
y = data_train['Survived']
x = data_train.drop('Survived', axis=1)


# Here split our data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.02, random_state=10)

# Init the model
model = LogisticRegression(random_state=0, max_iter=10000)

# train our model
model.fit(x_train, y_train)
predictions = model.predict(x_val)

# Great !!! our model is now finished and ready to use


# It's time to check the accuracy for our model
from sklearn.metrics import accuracy_score
print('Accuracy=', accuracy_score(y_val, predictions))

# Now we submit our model to kaggle
submit_pred = model.predict(data_test)
test = pd.read_csv('titanic/test.csv')
df = pd.DataFrame({'PassengerId': test['PassengerId'].values, 'Survived': submit_pred})
df.to_csv('first.csv', index=False)
