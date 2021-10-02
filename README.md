# -ML-Kaggle-Titanic-Problem-using-LogisticRegrission
here you will find the solution for the titanic problem on kaggle with comments and step by step coding

<br/>

<img src="https://storage.googleapis.com/kaggle-competitions/kaggle/3136/logos/header.png">

<br/>

## Problem Overview
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

<br/>

## <details open="open">
  <summary>Table of Contents</summary>
  <ol>
     <li>Analuze and visilaze the Dataset</li>
     <li>Clean and prepare the dataset for our ML model</li>
     <li>Build & Train Our Model</li>
     <li>Caluclate the Accuracy for the model</li>
     <li>Prepare the submission file to submit it to kaggle </li>
  </ol>
</details>

<br/>

### Load & Analyze Our Dataset
* First we read the data from the csv files
   ```py
   data_train = pd.read_csv('titanic/train.csv')
   data_test = pd.read_csv('titanic/test.csv')

   ```
  <br/>
  
### visilyze the given data
   ```py
      print(data_train.head())
   ```
   ```sh
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S   
```
<br/>
## Note
   ```sh
      The Survived column  is what we’re trying to predict. We call this column the (target) and remaining columns are called (features)
   ```
<br/>
### count the number of the Survived and the deaths
   ```py
   data_train['Survived'].value_counts()  # (342 Survived) | (549 not survived)
   ```

### plot the amount of the survived and the deaths
   ```py
   plt.figure(figsize=(5, 5))
   plt.bar(list(data_train['Survived'].value_counts().keys()), (list(data_train['Survived'].value_counts())),
        color=['r', 'g'])
   ```

<img src="https://github.com/elMeniwy/-ML-Kaggle-Titanic-Problem-using-LogisticRegrission/blob/main/images/deaths-survived.jpg">
  
### analyze the age
   ```py
plt.figure(figsize=(5, 7))
plt.hist(data_train['Age'], color='Purple')
plt.title('Age Distribuation')
plt.xlabel('Age')
plt.show()
   ```
<img src="https://github.com/elMeniwy/-ML-Kaggle-Titanic-Problem-using-LogisticRegrission/blob/main/images/age.png">
<br/>

### Note: Now after we made some analyze here and their, it's time to clean up our data If you take a look to the avalible columns we you may noticed that some columns are useless so they may affect on our model performance.

#### Here we make our cleaning function

```py
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
```

### # now we call our function and start cleaning!


```py
data_train = clean(data_train)
data_test = clean(data_test)
```
<br/>
## Note: now we need to change the sex feature into a numeric value like [1] for male and [0] female and also for the Embarked feature

### Here we used preprocessing method in sklearn to do this job
```py
le = preprocessing.LabelEncoder()
cols = ['Sex', 'Embarked'].predic
for col in cols:
    data_train[col] = le.fit_transform(data_train[col])
    data_test[col] = le.fit_transform(data_test[col])
 ```
<br/> 
## now our data is ready! it's time to build our model

### we select the target column ['Survived'] to store it in [Y] and drop it from the original data
```py
y = data_train['Survived']
x = data_train.drop('Survived', axis=1)
```


### Here split our data
```py
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.02, random_state=10)
```

### Init the model

```py
model = LogisticRegression(random_state=0, max_iter=10000)
```

### train our model
```py
model.fit(x_train, y_train)
predictions = model.predict(x_val)
```

<br/>
## Great !!! our model is now finished and ready to use

### It's time to check the accuracy for our model

```py
print('Accuracy=', accuracy_score(y_val, predictions))
```

Output:
```sh
Accuracy=0.97777
```


### Now we submit our model to kaggle

```py
test = pd.read_csv('titanic/test.csv')
df = pd.DataFrame({'PassengerId': test['PassengerId'].values, 'Survived': submit_pred})
df.to_csv('submit_this_file.csv', index=False)
```
















  
