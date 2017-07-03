from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
import pandas as pd


def get_data(filename):
    df = pd.read_csv(filename)

    # Fill in null values
    df.Age.fillna(df.Age.median(), inplace=True)
    df.Embarked.fillna(0, inplace=True)
    df.Cabin.fillna(0, inplace=True)
    df.Fare.fillna(0, inplace=True)
    return df


def get_vectorized_features(df, vectorizer=None):
    COLS = [
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Fare',
        'Cabin',
        'Embarked',
    ]

    data_dict = df[COLS].to_dict(orient='records')
    if not vectorizer:
        vec = DictVectorizer()
        vectorized = vec.fit_transform(data_dict)
        return vec,vectorized.toarray()
    else:
        vectorized = vectorizer.transform(data_dict)
        return vectorizer,vectorized.toarray()


def get_prediction(classifier='LinearSVC'):
    if classifier == 'LinearSVC':
        clf = svm.LinearSVC(loss='hinge')
    else:
        clf = RandomForestClassifier(n_estimators=20)
    print cross_val_score(clf, X, Y, cv=20).mean()

    clf.fit(X, Y)
    results = clf.predict(test_X)
    
    return results

train_df = get_data(filename='data/train.csv')
test_df = get_data(filename='data/test.csv')

vec,X = get_vectorized_features(train_df)
vec,test_X = get_vectorized_features(test_df,vec)

Y = train_df.Survived

results = get_prediction('RandomForestClassifier')
results_df = pd.DataFrame([test_df.PassengerId, results]).transpose()
results_df.columns = ['PassengerId', 'Survived']
results_df.to_csv('submissions/rfc_submission.csv', header=True, index=False)

results = get_prediction('LinearSVC')
results_df = pd.DataFrame([test_df.PassengerId, results]).transpose()
results_df.columns = ['PassengerId', 'Survived']
results_df.to_csv('submissions/linear_svc_submission.csv', header=True, index=False)
