from itertools import islice
from syscalls import syscalls
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

import numpy as np

import argparse
import os

WINDOW_SIZE = 0
N_NEIGHBORS = 3

LABEL_MULT_NORMAL = 0
LABEL_MULT_ANORMAL = 1

LABEL_ONE_NORMAL = 1
LABEL_ONE_ANORMAL = -1

RUNS = 10

FILES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "wordpress", "{v}", "{b}")


def sliding_window_filter(input_file):
    it = iter(input_file)
    result = ()
    for elem in it:
        if (elem.startswith("---")):
            elem = elem.split(" ")[1]
        if ("threat" in syscalls[elem.split("(")[0]]):
            if (syscalls[elem.split("(")[0]]["threat"] != 4):
                result = result + (syscalls[elem.split("(")[0]]["id"],)
        else:
            raise Exception(f"Threat para {elem.split('(')[0]} não encontrada")
        if len(result) == WINDOW_SIZE:
            yield result
            break
    for elem in it:
        if (elem.startswith("---")):
            elem = elem.split(" ")[1]
        if ("threat" in syscalls[elem.split("(")[0]]):
            if (syscalls[elem.split("(")[0]]["threat"] != 4):
                result = result[1:] + (syscalls[elem.split("(")[0]]["id"],)
                yield result
        else:
            raise Exception(f"Threat para {elem.split('(')[0]} não encontrada")


def sliding_window_raw(seq):
    it = iter(seq)
    result = tuple(syscalls[line.split(" ")[1] if line.startswith("---") else line.split("(")[0]]["id"] for line in islice(it, WINDOW_SIZE))
    if len(result) == WINDOW_SIZE:
        yield result
    for elem in it:
        if (elem.startswith("---")):
            elem = elem.split(" ")[1]
        result = result[1:] + (syscalls[elem.split("(")[0]]["id"],)
        yield result


def retrieve_dataset(filename, filter):

    with open(filename, "r") as input_file:
        if filter == "raw":
            dataset = list(sliding_window_raw(input_file))
        else:
            dataset = list(sliding_window_filter(input_file))

    return dataset


def define_labels(base_normal, base_exec, multi):
    labels = []

    label_normal = LABEL_MULT_NORMAL if multi else LABEL_ONE_NORMAL
    label_anormal = LABEL_MULT_ANORMAL if multi else LABEL_ONE_ANORMAL

    for window in base_normal:
        labels.append(label_normal)

    for window in base_exec:
        labels.append(label_anormal)

    return labels


def get_features(version, filter="raw"):

    path = FILES_PATH.format(v=version, b="normal")
    base_normal = []
    base_exec = []

    for file in os.listdir(path):
        base_normal.extend(retrieve_dataset(os.path.join(path, file), filter))

    path = FILES_PATH.format(v=version, b="exec")

    for file_exec in os.listdir(path):
        base_exec.extend(retrieve_dataset(os.path.join(path, file_exec), filter))

    return base_normal, base_exec


def naive_bayes(base_normal, base_exec):

    print("\n> Naive Bayes")

    results = []

    print("[...] Retrieving datasets and labels")
    labels = define_labels(base_normal, base_exec, True)
    features = base_normal + base_exec

    for i in range(RUNS):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=2**i)

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)

        score = (precision_score(y_test, y_pred, average="binary"), recall_score(y_test, y_pred, average="binary"), f1_score(y_test, y_pred, average="binary"), accuracy_score(y_test, y_pred))
        results.append(list(score))

    results = np.mean(results, axis=0)

    print("precision_score:", results[0])
    print("recall_score:", results[1])
    print("f1_score:", results[2])
    print("accuracy_score:", results[3])
    print("")

    return


def kneighbors(base_normal, base_exec):

    print("\n> K-Nearest Neighbors")

    results = []

    print("N_NEIGHBORS", str(N_NEIGHBORS))

    print("[...] Retrieving datasets and labels")
    # features,labels = get_features_labels(LABEL_MULT_NORMAL,LABEL_MULT_ANORMAL)
    labels = define_labels(base_normal, base_exec, True)
    features = base_normal + base_exec

    for i in range(RUNS):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=2**i)

        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=-1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        score = (precision_score(y_test, y_pred, average="binary"), recall_score(y_test, y_pred, average="binary"), f1_score(y_test, y_pred, average="binary"), accuracy_score(y_test, y_pred))
        results.append(list(score))

    results = np.mean(results, axis=0)

    print("precision_score:", results[0])
    print("recall_score:", results[1])
    print("f1_score:", results[2])
    print("accuracy_score:", results[3])
    print("")

    return


def random_forest(base_normal, base_exec):

    print("\n> Random Forest")

    results = []

    print("[...] Retrieving datasets and labels")
    labels = define_labels(base_normal, base_exec, True)
    features = base_normal + base_exec

    for i in range(RUNS):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=2**i)

        rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)

        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        score = (precision_score(y_test, y_pred, average="binary"), recall_score(y_test, y_pred, average="binary"), f1_score(y_test, y_pred, average="binary"), accuracy_score(y_test, y_pred))
        results.append(list(score))

    results = np.mean(results, axis=0)

    print("precision_score:", results[0])
    print("recall_score:", results[1])
    print("f1_score:", results[2])
    print("accuracy_score:", results[3])
    print("")

    return


def ada_boost(base_normal, base_exec):
    print("\n> Ada Boost")

    results = []

    print("[...] Retrieving datasets and labels")
    labels = define_labels(base_normal, base_exec, True)
    features = base_normal + base_exec

    for i in range(RUNS):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=2**i)

        abc = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_jobs=-1))
        abc.fit(X_train, y_train)
        y_pred = abc.predict(X_test)

        score = (precision_score(y_test, y_pred, average="binary"), recall_score(y_test, y_pred, average="binary"), f1_score(y_test, y_pred, average="binary"), accuracy_score(y_test, y_pred))
        results.append(list(score))

    results = np.mean(results, axis=0)

    print("precision_score:", results[0])
    print("recall_score:", results[1])
    print("f1_score:", results[2])
    print("accuracy_score:", results[3])
    print("")

    return


def multilayer_perceptron(base_normal, base_exec):
    print("\n> Multilayer Perceptron")

    results = []

    print("[...] Retrieving datasets and labels")
    labels = define_labels(base_normal, base_exec, True)
    features = base_normal + base_exec

    for i in range(RUNS):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=2**i)

        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        score = (precision_score(y_test, y_pred, average="binary"), recall_score(y_test, y_pred, average="binary"), f1_score(y_test, y_pred, average="binary"), accuracy_score(y_test, y_pred))
        results.append(list(score))

    results = np.mean(results, axis=0)

    print("precision_score:", results[0])
    print("recall_score:", results[1])
    print("f1_score:", results[2])
    print("accuracy_score:", results[3])
    print("")

    return


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
#     print("\nf1_score: ", f1_score(y_test, y_pred, average="binary"))
#     print("\nrecall_score: ", recall_score(y_test, y_pred, average="binary"))
#     print("\nprecision_score: ", precision_score(y_test, y_pred, average="binary"))
#     print("\n")
#
#     return lsvc

def one_class_svm(base_normal, base_exec):
    print("\n> One Class SVM")

    results = []

    print("[...] Retrieving datasets and labels")
    labels = define_labels(base_normal, base_exec, False)
    features = base_normal + base_exec

    for i in range(RUNS):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=2**i)

        onesvm = OneClassSVM(gamma="scale", nu=0.01)
        trainX = []
        for x, y in zip(X_train, y_train):
            if (y == 1):
                trainX.append(x)

        onesvm.fit(trainX)
        y_pred = onesvm.predict(X_test)

        score = (precision_score(y_test, y_pred, average="binary", pos_label=-1), recall_score(y_test, y_pred, average="binary", pos_label=-1), f1_score(y_test, y_pred, average="binary", pos_label=-1), accuracy_score(y_test, y_pred))
        results.append(list(score))

    results = np.mean(results, axis=0)

    print("precision_score:", results[0])
    print("recall_score:", results[1])
    print("f1_score:", results[2])
    print("accuracy_score:", results[3])
    print("")

    return


def isolation_forest(base_normal, base_exec):

    print("\n> Isolation Forest")

    results = []

    print("[...] Retrieving datasets and labels")
    labels = define_labels(base_normal, base_exec, False)
    features = base_normal + base_exec

    for i in range(RUNS):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=2**i)

        clf = IsolationForest(n_jobs=-1)
        trainX = []
        for x, y in zip(X_train, y_train):
            if (y == 1):
                trainX.append(x)

        clf.fit(trainX)
        y_pred = clf.predict(X_test)

        score = (precision_score(y_test, y_pred, average="binary", pos_label=-1), recall_score(y_test, y_pred, average="binary", pos_label=-1), f1_score(y_test, y_pred, average="binary", pos_label=-1), accuracy_score(y_test, y_pred))
        results.append(list(score))

    results = np.mean(results, axis=0)

    print("precision_score:", results[0])
    print("recall_score:", results[1])
    print("f1_score:", results[2])
    print("accuracy_score:", results[3])
    print("")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("window_size", help="Window size", type=int)
    parser.add_argument("-d", "--dataset", help="Dataset version to use", choices=["v1", "v2"], default="v2")
    parser.add_argument("-f", "--filter", help="Filter mode", choices=["raw", "filter"], default="raw")
    args = parser.parse_args()

    if args.window_size <= 0:
        raise argparse.ArgumentTypeError("window_size must be greater than 0")

    WINDOW_SIZE = args.window_size

    print(" ".join(("\n --- WINDOW_SIZE =", str(WINDOW_SIZE), "({}) --- \n".format(args.filter))))

    base_normal, base_exec = get_features(args.dataset, args.filter)

    naive_bayes(base_normal, base_exec)
    kneighbors(base_normal, base_exec)
    random_forest(base_normal, base_exec)
    multilayer_perceptron(base_normal, base_exec)
    ada_boost(base_normal, base_exec)

    one_class_svm(base_normal, base_exec)
    isolation_forest(base_normal, base_exec)
