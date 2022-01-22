"""Using Random Forest Classifier"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train.csv')

df['Male'] = df['Sex'] == 'Male'

df['Age'] = df['Age'].replace(np.nan, 0)

X = df[['Pclass','Age', 'SibSp', 'Parch', 'Fare', 'Male']].values
y = df['Survived'].values

rf = RandomForestClassifier(n_estimators=10, random_state=111)
rf.fit(X, y)
print(rf.score(X, y) * 100, '% Accuracy.')
ft_imp = pd.Series(rf.feature_importances_, index=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Male']).sort_values(ascending=False)
print('In order of importance: \n', ft_imp.head(10))


test = pd.read_csv('test.csv')
test['Male'] = test['Sex']=='male'
test['Age'] = test['Age'].replace(np.nan, 0)
test['Fare'] = test['Fare'].replace(np.nan, 0)
X_tes = test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Male']].values
print(rf.predict(X_tes))

ans = {'PassengerId' : test['PassengerId'], 'Survived' : rf.predict(X_tes)}
answer = pd.DataFrame(ans)
answer.set_index('PassengerId', inplace=True)
print(answer)
# answer.to_csv('Titanic Competition Submission.csv')
