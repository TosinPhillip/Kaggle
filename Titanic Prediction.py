"""Let's import what we need"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

'''Let's munge the dataset. Some values don't have an effect on the
survival.

1. Drop the PassengerId
2. Drop the Name
3. Drop Cabin. About 4 out of 5 don't have Cabin value 
4. Drop Embarked too.
5. At most 7 people have the same ticket, so it is unnecessary.
'''

df.drop('PassengerId', axis=1, inplace=True)
df.drop('Name', axis=1, inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df.drop('Embarked', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)

"""Let's convert some columns to floats using NumPy values method."""

df['Male'] = df['Sex']=='male'


"""Now drop Sex"""
df.drop('Sex', axis=1, inplace=True)

"""Fill every empty value with zero"""

df['Age'] = df['Age'].replace(np.nan, 0)
print(df.info())

model = LogisticRegression()

"""Let's prepare our feature set..."""

X = df[['Pclass','Age', 'SibSp', 'Parch', 'Fare', 'Male']].values

"""...and our target set. 1 means Survived while 0 means Deceased."""
y = df['Survived'].values


model.fit(X, y)
print(model.score(X, y))
'''
plt.scatter(X, y, c=df['Pclass'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend('True')
plt.show()'''
