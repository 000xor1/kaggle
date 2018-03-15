# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import setting as path


train = pd.read_csv(path.TRAIN)
test = pd.read_csv(path.TEST)
allData = [train, test]


sex_mapping = {"male": 0, "female": 1}

for dataset in allData:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


for dataset in allData:
        dataset['TitleName'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.',
                                                           expand=False)

titlename_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3,
                     "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,
                     "Countess": 4, "Ms": 4, "Lady": 4, "Jonkheer": 4,
                     "Don": 4, "Dona": 4, "Mme": 4, "Capt": 4, "Sir": 4}

for dataset in allData:
    dataset['TitleName'] = dataset['TitleName'].map(titlename_mapping)


train["Age"].fillna(train.groupby(["TitleName", "Sex"])
                    ["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby(["TitleName", "Sex"])
                   ["Age"].transform("median"), inplace=True)


for dataset in allData:
    dataset.loc[dataset['Age'] <= 18, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 36), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 2,
    dataset.loc[dataset['Age'] > 62, 'Age'] = 3


for dataset in allData:
    dataset['HasCabin'] = dataset['Cabin'].notnull().astype(int)


for dataset in allData:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2,
                 "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}


for dataset in allData:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6,
                  6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6,
                  11: 4}

for dataset in allData:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in allData:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


train["Embarked"].fillna(train.groupby("Pclass")["Embarked"].transform("median"),inplace=True)
test["Embarked"].fillna(train.groupby("Pclass")["Embarked"].transform("median"),inplace=True)


train["Fare"].fillna(train.groupby(["TitleName", "Sex"])
                     ["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby(["TitleName", "Sex"])
                    ["Fare"].transform("median"), inplace=True)


for dataset in allData:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 3


features_drop = ['Ticket', 'SibSp', 'Parch', 'Name']

train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)
train_data = train.drop('Survived', axis=1)
target = pd.DataFrame({"Survived": train['Survived']})


train_data.to_csv(path.TRAIN_X, index=False)
target.to_csv(path.TRAIN_Y, index=False)

test.to_csv(path.TEST_X, index=False)
