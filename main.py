from itertools import islice
from syscalls import syscalls
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import AdaBoostClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import argparse

WINDOW_SIZE         = 0
N_NEIGHBORS         = 3

LABEL_MULT_NORMAL   = 0
LABEL_MULT_ANORMAL  = 1

LABEL_ONE_NORMAL    = 1
LABEL_ONE_ANORMAL   = -1

BASE_NORMAL         = 'wordpress/v1/wordpress_normal_1'
BASE_EXEC           = 'wordpress/v1/wordpress_exec_1_teste1'
# BASE_EXEC_TESTE     = 'wordpress_exec_1_teste2'

def sliding_window_filter(input_file):
    it = iter(input_file)
    result = ()
    for elem in it:
        if (elem.startswith("---")):
            elem = elem.split(' ')[1]
        if ("threat" in syscalls[elem.split('(')[0]]):
            if (syscalls[elem.split('(')[0]]['threat'] != 4):
                result = result + (syscalls[elem.split('(')[0]]['id'],)
        else:
            print(f'Threat para {elem} não encontrada')
            return
        if len(result) == WINDOW_SIZE:
            yield result
            break
    for elem in it:
        if (elem.startswith("---")):
            elem = elem.split(' ')[1]
        if ("threat" in syscalls[elem.split('(')[0]]):
            if (syscalls[elem.split('(')[0]]['threat'] != 4):
                result = result[1:] + (syscalls[elem.split('(')[0]]['id'],)
                yield result
        else:
            print(f'Threat para {elem} não encontrada')
            return

def sliding_window_raw(seq):
    it = iter(seq)
    result = tuple(syscalls[line.split(' ')[1] if line.startswith("---") else line.split('(')[0]]['id'] for line in islice(it,WINDOW_SIZE))
    if len(result) == WINDOW_SIZE:
        yield result
    for elem in it:
        if (elem.startswith("---")):
            elem = elem.split(' ')[1]
        result = result[1:] + (syscalls[elem.split('(')[0]]['id'],)
        yield result

def retrieve_dataset(filename):
    with open(filename, 'r') as input_file:
        dataset = list(sliding_window_raw(input_file));

    return dataset

def define_labels(base_normal,base_exec,label_normal,label_anormal):
    labels = []

    for window in base_normal:
        labels.append(label_normal)

    for window in base_exec:
        labels.append(label_anormal)

    return labels

def get_features_labels(label_normal,label_anormal):

    labels = []

    # Base file
    base_normal = retrieve_dataset(BASE_NORMAL)

    base_exec = retrieve_dataset(BASE_EXEC)

    labels = define_labels(base_normal,base_exec,label_normal,label_anormal)

    features = base_normal + base_exec

    return features,labels

# def forrest_alg():
#
#     print("\n> STIDE")
#
#     anom_count = 0;
#     win_count = 0;
#
#     base_normal = retrieve_dataset(BASE_NORMAL)
#     # print(base_normal)
#
#     # Exec file -- forrest alg.
#
#     with open(BASE_EXEC, 'r') as input_file:
#         for elem in sliding_window_raw(input_file):
#             win_count += 1;
#             if (elem not in base_normal):
#                 anom_count += 1
#                 # print(elem);
#
#     print('\nNumero de windows da base normal: ', len(base_normal))
#     print('\nNumero de windows da base exec: ', win_count)
#     print('\nNumero de anomalias detectadas: ', anom_count)
#     print('\nTaxa de detecção ', anom_count/win_count )

def naive_bayes():

    print("\n> Naive Bayes")

    print("\n[...] Retrieving datasets and labels")
    features,labels = get_features_labels(LABEL_MULT_NORMAL,LABEL_MULT_ANORMAL)

    X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

    gnb = GaussianNB()

    print("\n[...] Training the model")
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)


    # base_teste = retrieve_dataset(BASE_EXEC_TESTE)
    # labels_teste = []
    #
    # for elem in base_teste:
    #     labels_teste.append(1)

    # predict = gnb.predict(base_teste)

    # print("\nWindows base normal: ",len_normal)
    # print("\nWindows base exec: ",len_exec)
    # print("\nWindows base teste: ",len(base_teste))
    # print("\nAccuracy: ", metrics.accuracy_score(labels_teste,predict))

    print("\nf1_score: ", f1_score(y_test, y_pred, average='binary'))
    print("\nrecall_score: ", recall_score(y_test, y_pred, average='binary'))
    print("\nprecision_score: ", precision_score(y_test, y_pred, average='binary'))
    print("\n")

    return gnb

def kneighbors():

    print("\n> K-Nearest Neighbors")

    print('\nN_NEIGHBORS ' + str(N_NEIGHBORS))

    print("\n[...] Retrieving datasets and labels")
    features,labels = get_features_labels(LABEL_MULT_NORMAL,LABEL_MULT_ANORMAL)

    X_train,X_test,y_train,y_test = train_test_split(features,labels, test_size=0.5, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)

    print("\n[...] Training the model")
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("\nf1_score: ", f1_score(y_test, y_pred, average='binary'))
    print("\nrecall_score: ", recall_score(y_test, y_pred, average='binary'))
    print("\nprecision_score: ", precision_score(y_test, y_pred, average='binary'))
    print("\n")

    return knn

def random_forest():

    print("\n> Random Forest")

    print("\n[...] Retrieving datasets and labels")
    features,labels = get_features_labels(LABEL_MULT_NORMAL,LABEL_MULT_ANORMAL)

    X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

    rfc = RandomForestClassifier(n_estimators=100)

    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    print("\nf1_score: ", f1_score(y_test, y_pred, average='binary'))
    print("\nrecall_score: ", recall_score(y_test, y_pred, average='binary'))
    print("\nprecision_score: ", precision_score(y_test, y_pred, average='binary'))
    print("\n")

    return rfc

def ada_boost():
    print("\n> Ada Boost")

    print("\n[...] Retrieving datasets and labels")
    features,labels = get_features_labels(LABEL_MULT_NORMAL,LABEL_MULT_ANORMAL)

    X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

    abc = AdaBoostClassifier(base_estimator=RandomForestClassifier())

    abc.fit(X_train, y_train)
    y_pred = abc.predict(X_test)

    print("\nf1_score: ", f1_score(y_test, y_pred, average='binary'))
    print("\nrecall_score: ", recall_score(y_test, y_pred, average='binary'))
    print("\nprecision_score: ", precision_score(y_test, y_pred, average='binary'))
    print("\n")

    return abc

def multilayer_perceptron():
    print("\n> Multilayer Perceptron")

    print("\n[...] Retrieving datasets and labels")
    features,labels = get_features_labels(LABEL_MULT_NORMAL,LABEL_MULT_ANORMAL)

    X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

    mlp = MLPClassifier()

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    print("\nf1_score: ", f1_score(y_test, y_pred, average='binary'))
    print("\nrecall_score: ", recall_score(y_test, y_pred, average='binary'))
    print("\nprecision_score: ", precision_score(y_test, y_pred, average='binary'))
    print("\n")

    return mlp


# def linear_svc():
#     print("\n> Linear SVC")
#
#     print("\n[...] Retrieving datasets and labels")
#     features,labels = get_features_labels()
#
#     X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.5, random_state=42)
#
#     lsvc = SVC()
#
#     lsvc.fit(X_train, y_train)
#     y_pred = lsvc.predict(X_test)
#
#     print("\nf1_score: ", f1_score(y_test, y_pred, average='binary'))
#     print("\nrecall_score: ", recall_score(y_test, y_pred, average='binary'))
#     print("\nprecision_score: ", precision_score(y_test, y_pred, average='binary'))
#     print("\n")
#
#     return lsvc

def one_class_svm():
    print("\n> One Class SVM")

    print("\n[...] Retrieving datasets and labels")
    features,labels = get_features_labels(LABEL_ONE_NORMAL,LABEL_ONE_ANORMAL)

    X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

    onesvm = OneClassSVM(gamma='scale', nu=0.01)
    # print("\n[...] Training model")
    trainX = []

    for x,y in zip(X_train, y_train):
        if (y == 1):
            trainX.append(x)

    onesvm.fit(trainX)
    y_pred = onesvm.predict(X_test)

    print("\nf1_score: ", f1_score(y_test, y_pred, average='binary'))
    print("\nrecall_score: ", recall_score(y_test, y_pred, average='binary'))
    print("\nprecision_score: ", precision_score(y_test, y_pred, average='binary'))
    print("\n")

    return onesvm

def isolation_forest():

    print("\n> Isolation Forest")

    print("\n[...] Retrieving datasets and labels")
    features,labels = get_features_labels(LABEL_ONE_NORMAL,LABEL_ONE_ANORMAL)

    X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

    clf = IsolationForest()

    trainX = []

    for x,y in zip(X_train, y_train):
        if (y == 1):
            trainX.append(x)

    clf.fit(trainX)
    y_pred = clf.predict(X_test)

    print("\nf1_score: ", f1_score(y_test, y_pred, average='binary'))
    print("\nrecall_score: ", recall_score(y_test, y_pred, average='binary'))
    print("\nprecision_score: ", precision_score(y_test, y_pred, average='binary'))
    print("\n")

    return clf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('window_size', help='window size')
    args = parser.parse_args()

    # WINDOW_SIZE = int(1628 * float(args.percentage))
    # print('\n --- WINDOW_SIZE ' + str(int((float(args.percentage) * 100))) + '% = ' + str(WINDOW_SIZE) + ' --- \n')

    WINDOW_SIZE = int(args.window_size)

    print('\n --- WINDOW_SIZE = ' + str(WINDOW_SIZE) + ' --- \n')

    # naive_bayes()
    #kneighbors()
    #random_forest()
    #multilayer_perceptron()
    #ada_boost()

    one_class_svm()
    # isolation_forest()
