import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score
from IPython.display import display


# Разведочный анализ данных

def heatmap(data):
    """Карта корреляций"""
    plt.figure()
    sns.heatmap(data.corr(), annot=True)
    plt.title("Карта корреляций")
    plt.show()

def balanced_sample(target):
    """Сбалансированность выборки"""
    temp = set(target)
    classes_size = {}
    for i in temp:
        classes_size[i] = list(target).count(i)
    print(classes_size)

def pca(data, target):
    plt.figure()
    data_r_2 = PCA(n_components=2)#, random_state=0)
    data_reduced_2 = data_r_2.fit_transform(data)
    plt.scatter(data_reduced_2[:, 0], data_reduced_2[:, 1], c=target)
    fig = plt.figure()
    ax = Axes3D(fig)
    data_r_3 = PCA(n_components=3)#, random_state=0)
    data_reduced_3 = data_r_3.fit_transform(data)
    ax.scatter(data_reduced_3[:, 0], data_reduced_3[:, 1], data_reduced_3[:, 2], s=40, c=target)
    ax.set_title('Visualization')
    print("Выделенные компоненты", data_r_2.components_, "Главные компоненты(2)", data_r_2.explained_variance_ratio_, sep='\n')
    print("Главные компоненты(3)", data_r_3.explained_variance_ratio_, sep='\n')
    #data_r_mle = PCA(n_components='mle')
    #print("Главные компоненты(mle)", data_r_mle.explained_variance_ratio_, sep='\n')
    plt.show()


def tsne(data, target, pca=False):
    plt.figure()
    if pca:
        data_reduced_2 = TSNE(n_components=2, init='pca').fit_transform(data)
    else:
        data_reduced_2 = TSNE(n_components=2).fit_transform(data)
    plt.scatter(data_reduced_2[:, 0], data_reduced_2[:, 1], c=target)
    fig = plt.figure()
    ax = Axes3D(fig)
    if pca:
        data_reduced_3 = TSNE(n_components=3).fit_transform(data)
    else:
        data_reduced_3 = TSNE(n_components=3).fit_transform(data)
    ax.scatter(data_reduced_3[:, 0], data_reduced_3[:, 1], data_reduced_3[:, 2], s=30, c=target)
    ax.set_title('Visualization')
    plt.show()


# Классификация

def logistic_KNeighbors(data, target):
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    Y = model.predict(X_test)
    Y_proba = model.predict_proba(X_test)
    print(Y)
    print("classification_report", classification_report(Y_test, Y), sep='\n')
    print("confusion_matrix", confusion_matrix(Y_test, Y), sep='\n')
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(X_test)
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=Y_test)
    plt.title('x test, y test, Исходные метки')
    plt.show()
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=Y)
    plt.title('x test, y predict Логистическая регрессия')
    plt.show()
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    Y = model.predict(X_test)
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(X_test)
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=Y)
    plt.title('x test, y predict метод K-соседей')
    plt.show()
    print("classification_report", classification_report(Y_test, Y), sep='\n')
    print("confusion_matrix", confusion_matrix(Y_test, Y), sep='\n')

def search_parameters(data, target):
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3)
    model = GridSearchCV(KNeighborsClassifier(),
        {'n_neighbors': list(range(2, 6)), 'weights': ['uniform', 'distance']}, n_jobs=3, cv=5)
    print("Обучение модели")
    model.fit(X_train, Y_train)
    print('Best_params')
    print(model.best_params_)
    print('Report')
    print(classification_report(Y_test, model.predict(X_test)))
    model2 = GridSearchCV(LogisticRegression(),
                        {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                         'penalty': ['l1', 'l2'],
                         'C': [i / 10 for i in range(1,15)]},
                         n_jobs=3, cv=5)
    """ print('Обучение модели')
    model2.fit(X_train, Y_train)
    print('Best params')
    print(model2.best_params_)
    print('Report')
    print(classification_report(Y_test, model.predict(X_test))) """


def logistic_KNeighbors_with_parameters(data, target, n_neighbors, weights, C, penalty, solver):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(x_train, y_train)
    y = model.predict(x_test)
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(x_test)
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=y)
    plt.title('x test, y predict K-ближайших соседей')
    plt.show()
    print(classification_report(y_test, y))
    print(confusion_matrix(y_test, y))
    model2 = LogisticRegression(C=C, penalty=penalty, solver=solver)
    model2.fit(x_train, y_train)
    y = model2.predict(x_test)
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(x_test)
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=y)
    plt.title('x test, y predict, Логистическая регрессия')
    plt.show()
    print(classification_report(y_test, y))
    print(confusion_matrix(y_test, y))


def random_forest(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.30)
    model = GridSearchCV(RandomForestClassifier(),
                         {'criterion': ["gini", "entropy"],
                          'n_estimators':list(range(10,101,10)),
                          'max_depth': list(range(10,151,10)),
                          'max_features': list(range(10,151,10))},
                         n_jobs=3, cv=5)

    model.fit(X_train, y_train)


    y_proba = model.predict(X_test)

    plt.figure()
    X_reduced_2 = PCA(n_components=2).fit_transform(X_test, y_test)
    plt.scatter(X_reduced_2[:, 0], X_reduced_2[:, 1],
                c=y_test,
                cmap=plt.cm.Set1,
                edgecolor="k",
                s=40,
                )
    plt.title('С исходными метками')

    plt.figure()
    Y_reduced_2 = PCA(n_components=2).fit_transform(X_test, y_proba)
    plt.scatter(Y_reduced_2[:, 0], Y_reduced_2[:, 1],
                c=y_proba,
                cmap=plt.cm.Set1,
                edgecolor="k",
                s=40,
                )
    plt.title('Random Forest,С метками полученными при классификации')
    print('Обучение модели')
    print('Best_params')
    print(model.best_params_)
    print('Report')
    print(classification_report(y_test, model.predict(X_test)))

    print('Оценка точности-', accuracy_score(y_proba, y_test), '\n')
    print('Mатрица неточности:\n ', confusion_matrix(y_proba, y_test))

    plt.show()
    
def random_tree(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.30)
    model = GridSearchCV(DecisionTreeClassifier(),
                         {'criterion': ["gini", "entropy"],
                          'max_depth': list(range(100,201,10)),
                          'max_features': list(range(0,101,10))},
                         n_jobs=3, cv=5)
    model.fit(X_train, y_train)

    y_proba = model.predict(X_test)

    plt.figure()
    X_reduced_2 = PCA(n_components=2).fit_transform(X_test, y_test)
    plt.scatter(X_reduced_2[:, 0], X_reduced_2[:, 1],
                c=y_test,
                cmap=plt.cm.Set1,
                edgecolor="k",
                s=40,
                )
    plt.title('С исходными метками')

    plt.figure()
    Y_reduced_2 = PCA(n_components=2).fit_transform(X_test, y_proba)
    plt.scatter(Y_reduced_2[:, 0], Y_reduced_2[:, 1],
                c=y_proba,
                cmap=plt.cm.Set1,
                edgecolor="k",
                s=40,
                )
    plt.title('Tree Classifier, С метками полученными при классификации')
    print('Обучение модели')
    print('Best_params')
    print(model.best_params_)
    print('Report')
    print(classification_report(y_test, model.predict(X_test)))
    print('Оценка точности-', accuracy_score(y_proba, y_test), '\n')
    print('Mатрица неточности:\n ', confusion_matrix(y_proba, y_test))
    plt.show()



def main():
    data_read = pd.read_excel("Wine.xlsx")
    data = data_read.drop(columns="Wine")
    target = np.array(data_read[data_read.columns[0]], np.int32)

    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_normalized = pd.DataFrame(x_scaled)    # Нормализованная выборка
    
    # Разведочный анализ
    #print(data_normalized)
    #print(data_normalized.describe())   # основные статистические данные
    #balanced_sample(target)
    #heatmap(data)
    #pca(data=data_normalized, target=target)
    #tsne(data=data_normalized, target=target, pca=False)
    #tsne(data=data_normalized, target=target, pca=True)

    # Классификация
    #logistic_KNeighbors(data_normalized, target)   # этот пункт в отчет не вставил
    #search_parameters(data_normalized, target)
    logistic_KNeighbors_with_parameters(data=data_normalized, target=target, n_neighbors=3, weights='uniform', C=0.1, penalty='l2', solver='newton-cg')
    #random_tree(data=data_normalized, target=target)
    #random_forest(data=data_normalized, target=target)


if __name__ == '__main__':
    main()