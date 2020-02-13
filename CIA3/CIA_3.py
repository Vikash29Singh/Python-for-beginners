# 1.Naive bayes classification
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('diabetes.csv')
print(data.head())
print(data.describe())
print(data.isna())
print(data.isna().sum())
x = data.drop('Outcome', axis=1)
print(data.head())
y = data['Outcome']
print(y.head())

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=100)

nb = GaussianNB()
nb.fit(xtrain, ytrain)

ypred = nb.predict(xtest)
print(ypred)

print("Accuracy:", metrics.accuracy_score(ytest, ypred))
print(metrics.classification_report(ytest, ypred))
print(metrics.confusion_matrix(ytest, ypred))
# -----------------------------------------------------------------------------------------------
'''
# 2.Rule based classification
import pandas as pd
from rulefit import RuleFit

redwine_data = pd.read_csv("diabetes.csv", index_col=0)
print(redwine_data.describe())
y = redwine_data.Outcome.values
X = redwine_data.drop("Outcome", axis=1)

features = X.columns
X = X.as_matrix()

rf = RuleFit()
rf.fit(X, y, feature_names=features)

rf.predict(X)
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values("support", ascending=False)
print(rules)

###############################################################

from skrules import SkopeRules
import pandas as pd

dataset_1 = pd.read_csv('diabetes.csv')
dataset = pd.DataFrame(dataset_1)

print(dataset)
print(dataset.describe())
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin']

clf = SkopeRules(max_depth_duplication=2,
                 n_estimators=30,
                 precision_min=0.3,
                 recall_min=0.1,
                 feature_names=feature_names)

for idx, Outcome in enumerate(dataset.feature_names):
    X, y = dataset.data, dataset.target
    clf.fit(X, y == idx)
    rules = clf.rules_[0:3]
    print("Rules for iris", Outcome)
    for rule in rules:
        print(rule)
    print()
    print(20 * '=')
    print()

###############################################################

from sklearn.datasets import load_boston
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from skrules import SkopeRules
import seaborn as sns

dataset = load_boston()
clf = SkopeRules(max_depth_duplication=None,
                 n_estimators=30,
                 precision_min=0.2,
                 recall_min=0.01,
                 feature_names=dataset.feature_names)

X, y = dataset.data, dataset.target > 25
X_train, y_train = X[:len(y) // 2], y[:len(y) // 2]
X_test, y_test = X[len(y) // 2:], y[len(y) // 2:]
clf.fit(X_train, y_train)
y_score = clf.score_top_rules(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_score)
print(recall)
print("Precision", precision)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve')
plt.show()
ax = sns.barplot(recall, precision)
ax.set(xlabel='Recall', ylabel='Precision')
plt.show()
'''
# -----------------------------------------------------------------------------------------------
# 3.K means clustering

from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('dermatology.data')
df.head(10)
print(df.describe())
x = df.iloc[:, 6:20].values

kmeans5 = KMeans(n_clusters=4)
y_kmeans5 = kmeans5.fit_predict(x)
print(y_kmeans5)

kmeans5.cluster_centers_

Error = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()

kmeans3 = KMeans(n_clusters=4)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)

kmeans3.cluster_centers_

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans3, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='blue')

###############################################################

from sklearn import datasets

iris = datasets.load_iris()
x, y = iris.data, iris.target

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x_reduced = pca.fit_transform(x)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

kmeans.fit(x)
ypred = kmeans.predict(x)

import matplotlib.pyplot as plt

plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=ypred, cmap='rainbow', alpha=0.5)
'''
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
'''
plt.show()

###########------------
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)