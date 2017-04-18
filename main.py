from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.svm import SVC, OneClassSVM, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sys import exit
from itertools import chain, combinations
from copy import deepcopy
import numpy as np
from math import isnan
from random import shuffle

class APriori():
      def __init__(self):
            self.rules = []
            self.minSupport, self.minConfidence = -1, -1
            self.distinctItems = []
      def addData(self, d):
            self.data = deepcopy(d)
      def preprocess():
            print("not yet")
      def run(self):
            if len(self.distinctItems) == 0:
                  preprocess()
            if self.minSupport == -1:
                  self.minSupport = len(self.distinctItems)//5
            if self.minConfidence == -1:
                  self.minConfidence = .6

class WeightedTrainingNeuralNet():
      def __init__(self, iterations=10000):
            self.n = iterations

      def printMe(self):
            print(self.synaptic_weights)
            print(self.n)

      def score(self, data):
            print("Need to find a way to implement the SCORE method in the others")
            exit()

      def fit(self, data, classes, weights):
            def mult(a,b):
                  return a * b

            # transform the data so that there are weights on each parameter
            training_set_inputs = np.array(data)
            training_set_outputs = np.array(classes).T
            training_set_weights = np.array(weights)
            self.synaptic_weights = np.random.random((training_set_inputs.shape[0],1))

            for iteration in range(self.n):
                output = 1 / (1 + np.exp(-(np.dot(training_set_inputs.T, self.synaptic_weights))))
                vfunc = np.vectorize(mult)
                print(output.shape)
                print(training_set_inputs.shape)
                print(training_set_weights.shape)
                exit()
                # then need to dot in a vector of weights
                self.synaptic_weights += np.dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output) * 1)

      def predict(self, test_inputs):
            predictions = []
            for t in test_inputs:
                  predictions.append(1 / (1 + np.exp(-(np.dot(np.array(t), self.synaptic_weights)))))
            return predictions

def avg(lst):
      return (sum(lst)+0.0)/len(lst)

def powerset(iterable):
    "from itertools documentation"
    s = list(iterable)
    return [''.join(r) for r in chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]

def clustering(X):
      X = StandardScaler().fit_transform(X)
      n = 25

      # DBSCAN
      db = DBSCAN(eps=5, min_samples=25).fit(X)
      core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
      core_samples_mask[db.core_sample_indices_] = True
      labels = db.labels_
      print(labels)

      # Number of clusters in labels, ignoring noise if present.
      n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

      print('Estimated number of clusters: %d' % n_clusters_)
      print("Calinski Harabaz: %0.3f" % metrics.calinski_harabaz_score(X, labels))
      print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))
      print('')


      #KMEANS
      kmeans = KMeans(n_clusters=n, random_state=0, init='k-means++').fit(X)
      print(kmeans.labels_)
      seen = {}
      for k in kmeans.labels_:
            if k in seen:
                  seen[k] += 1
            else:
                  seen[k] = 1
      print(seen)
      #print(kmeans.cluster_centers_)

      # Agglomerative Clustering
      agg = AgglomerativeClustering(n_clusters=n)
      agg.fit(X)
      print(agg.labels_)
      seen2 = {}
      for k in agg.labels_:
            if k in seen2:
                  seen2[k] += 1
            else:
                  seen2[k] = 1
      print(seen2)


def projectClassification(X, meta, headers=None, weights=None, verbose=False):
      kf = KFold(n_splits=10, shuffle=True)
      results = {}

      for i in range(len(headers)):
            print(i)
            XX = []
            yy = []
            results[headers[i]] = []

            # variable j is now the class variable.
            for line in X:
                  XX.append(line[:i]+line[i+1:])
                  yy.append(line[i])
            print("Class variable assigned")

            for train_index, test_index in kf.split(XX):
                  print("running experiment")
                  X_train, X_test = [XX[k] for k in train_index], [XX[j] for j in test_index]
                  y_train, y_test = [yy[k] for k in train_index], [yy[j] for j in test_index]
                  trainWeights = [w[0] if headers[w[0]] in headers[w[1]] else w[1] for w in weights]
                  a, b = classifiers(X_train, y_train, X_test, y_test, meta[i], trainWeights, verbose)
                  results[headers[i]].append((a,b))

            # return anything that gets better than X% accuracy
      print(results)
      with open('./results/results.txt', 'w') as cout:
            for r in results.keys():
                  cout.write(str(r)+":"+ str(results[r]))

      summary = {}
      for r in results.keys():
            mx = max(results[r], key=lambda item:item[0])
            mn = min(results[r], key=lambda item:item[0])
            summary[r] =(mn, mx)


      for s in summary.keys():
            print(s, summary[s])

"""
def classification(training, classification, tests, actuals, outFile):
      #Support Vector Machines
      svm = SVC()
      svm.fit(training, classification)

      # Naive Bayes
      nb = GaussianNB()
      nb.fit(training, classification)

      #K-Nearest Neighbors
      knn = KNeighborsClassifier()
      knn.fit(training, classification)

      # Random Forest
      randFor = RandomForestClassifier()
      randFor.fit(training, classification)

      # AdaBoost
      ada = AdaBoostClassifier()
      ada.fit(training, classification)

      # Neural Net (Backward Propogation)
      neuralnet = MLPClassifier()
      neuralnet.fit(training, classification)

      methods = [svm, nb, knn, randFor, ada, neuralnet]
      results = {m.__class__.__name__:[] for m in methods}
      index = {p: l for l,p in enumerate(powerset(results.keys())) if p != ''}
      #print([k for k in index.keys() if index[k] == 27])

      for m in methods:
            results[m.__class__.__name__] = m.predict(tests)

      for e in range(len(tests)):
            minVal = min([results[t][e] for t in results.keys()])
            tmp = [t for t in results.keys() if results[t][e] == minVal]
            tmp2 = tests[e]+[index[''.join(tmp)]]
            outFile.write(str(tmp2).strip('[]')+"\n")
"""


def outlierPreprocessing(fpath, d, splits=25):
    kf = KFold(n_splits=splits, shuffle=True)
    XX, yy = readInput(fpath+d+".txt")

    totesFeatures = []
    totesClasses = []


    for train_index, test_index in kf.split(XX):
        print("start")
        training = [XX[i] for i in train_index]

        ocs = OneClassSVM()
        ocs.fit(training)
        ocsOut = ocs.predict(training)

        one = [i for i,x in enumerate(ocsOut) if x == 1]
        two = [i for i,x in enumerate(ocsOut) if x == -1]
        totesFeatures.extend([XX[i] for i in two])
        totesClasses.extend(yy[i] for i in two)

        indexes = sample(one, 3*len(two))
        totesFeatures.extend([XX[i] for i in indexes])
        totesClasses.extend([yy[i] for i in indexes])
    return totesFeatures, totesClasses

def outlierDetection(fpath, d, splits=25):
      kf = KFold(n_splits=splits, shuffle=True)
      XX, yy = readInput(fpath+d+".txt")
      results = {'training':[], 'testing':[]}
      totesFeatures = []
      totesClasses = []


      #method = [isoFor, ocs]

      #for m in method:

      for train_index, test_index in kf.split(XX):
            print("start")
            training, tests = [XX[i] for i in train_index], [XX[j] for j in test_index]
            classes, actuals = [yy[i] for i in train_index], [yy[j] for j in test_index]

            # Random Forest
            #isoFor = IsolationForest()
            #isoFor.fit(training)
            #isoForOut = isoFor.predict(training if testing is None else testing)
            #isoForOut = isoFor.predict(training)
            #isoForTest = isoFor.predict(tests)


            ocs = OneClassSVM()
            ocs.fit(training)
            # #ocsOut = ocs.predict(training if testing is None else testing)
            ocsOut = ocs.predict(training)
            ocsTest = ocs.predict(tests)


            #one = [i for i,x in enumerate(isoForOut) if x == -1]

            # one = [i for i,x in enumerate(ocsOut) if x == 1]
            # two = [i for i,x in enumerate(ocsOut) if x == -1]
            #
            # oneTest = [i for i,x in enumerate(ocsTest) if x == 1]
            # twoTest = [i for i,x in enumerate(ocsTest) if x == -1]


            one = [i for i,x in enumerate(isoForOut) if x == 1]
            two = [i for i,x in enumerate(isoForOut) if x == -1]
            totesFeatures.extend([XX[i] for i in two])
            totesClasses.extend(yy[i] for i in two)

            #oneTest = [i for i,x in enumerate(isoForTest) if x == 1]
            #twoTest = [i for i,x in enumerate(isoForTest) if x == -1]


            #three = [i for i in one if i in two]
            #print(avg([1 if classes[i] > 0.5 else 0 for i in one]),len([1 if classes[i] > 0.5 else 0 for i in one]))
            ones = sum([1 if classes[i] <= 0.5 else 0 for i in one])
            twos = sum([1 if classes[i] > 0.5 else 0 for i in two])
            total = len(one) + len(two)
            results['training'].append([((ones + twos + 0.0)/total), ones, twos, total])

            onesTest = sum([1 if actuals[i] <= 0.5 else 0 for i in oneTest])
            twosTest = sum([1 if actuals[i] > 0.5 else 0 for i in twoTest])
            totalTest = len(oneTest) + len(twoTest)
            results['testing'].append([(onesTest + twosTest + 0.0)/totalTest, onesTest, twosTest, totalTest])
            with open('./results/outlier.txt', 'w') as cout:
                cout.write(results)

                #print(avg([1 if classes[i] > 0.5 else 0 for i in two]),len([1 if classes[i] > 0.5 else 0 for i in two]))
                #print(avg([1 if classes[i] > 0.5 else 0 for i in three]),len([1 if classes[i] > 0.5 else 0 for i in three]))
                #print(avg([1 if i > 0.5 else 0 for i in classes]),len([1 if i > 0.5 else 0 for i in classes]))
                #print("\n\n")

def classifiers(training, classes, tests, actuals, cts, trainWeights, verbose):

      if not cts: # discrete
             #Support Vector Machines
            svm = SVC()
            try:
                  svm.fit(training, classes)
            except ValueError:
                  distinct = []
                  for a in actuals:
                        if a not in distinct:
                              distinct.append(a)
                  print(len(distinct), len(actuals))
                  print(actuals)
                  exit()
            if verbose:
                  print("SVM done")

            # Naive Bayes
            nb = GaussianNB()
            nb.fit(training, classes)
            if verbose:
                  print("NB done")

            #K-Nearest Neighbors
            knn = KNeighborsClassifier()
            knn.fit(training, classes)
            if verbose:
                  print("KNN done")

            # Random Forest
            randFor = RandomForestClassifier()
            randFor.fit(training, classes)
            if verbose:
                  print("RandFor done")

            # AdaBoost
            ada = AdaBoostClassifier()
            ada.fit(training, classes)
            if verbose:
                  print("Ada done")

            # Neural Net (Backward Propogation)
            neuralnet = MLPClassifier()
            neuralnet.fit(training, classes)
            if verbose:
                  print("NeuralNetwork done")

            weightedNN = WeightedTrainingNeuralNet()
            # get weighted labels
            weightedNN.fit(training, classes, trainWeights)
            if verbose:
                  print("Weighted-Training Neural Network done")
            weightedNN.printMe()
            exit()

            methods = [svm, nb, knn, randFor, ada, neuralnet, weightedNN]
      else:
            # LASSO
            lassocv = LassoCV().fit(training, classes)
            if verbose:
                  print("LASSO done")

            # Polynomial Regression (up to degree 3)
            polyReg = PolynomialFeatures(degree=3).fit_transform(training)
            polyLasso = Lasso().fit(polyReg, classes)
            if verbose:
                  print("Polynomial Regression done")

            # SVR
            svr = SVR(C=1.0, epsilon=0.2).fit(training, classes)
            if verbose:
                  print("SVR done")

            methods = [lassocv, polyLasso, svr]


      results = {m:[] for m in methods}

      info = {}

      for m in methods:
            if cts:
                  if m != polyLasso:
                        info[m] = m.score(tests, actuals)
                  else:
                        info[m] = m.score(PolynomialFeatures(degree=3).fit_transform(tests), actuals)
            else:
                  info[m] = m.score(tests, actuals)

      maxVal = deepcopy(info[info.keys()[0]])
      maxMeth = deepcopy(info.keys()[0])
      for i in info.keys():
            if info[i] > maxVal:
                  maxVal = deepcopy(info[i])
                  maxMeth = deepcopy(i)

      return maxVal, maxMeth


      #C = sklearn.metrics.auc(true, predicted)

def dataMiningBonusHW1(training, classes, test, actual):
      svm = SVC()
      nb = GaussianNB()
      randFor = RandomForestClassifier()
      neuralnet = MLPClassifier()
      decTree = DecisionTreeClassifier()

      #methods = [svm, nb, randFor, neuralnet, decTree, mlc]
      #methods = [svm, nb, randFor, neuralnet, decTree]
      results = {m.__class__.__name__:[] for m in methods}
      results['majority'] = []

      #Support Vector Machines
      #svm.fit(training, classes)

      # Naive Bayes
      #nb.fit(training, classes)

      # Random Forest
      #randFor.fit(training, classes)

      # Neural Net (Backward Propogation)
      #neuralnet.fit(training, classes)

      # Decision Tree
      #decTree.fit(training, classes)
      toto = []
      for t in range(len(training)):
            toto.append(training[t] + classes[t])
      # Maximum Likelihood Classifier
      mlc = spy.GaussianClassifier(toto)

      """
      majority = [{} for _ in test]
      print(type(majority))
      for m in methods:
            tmp = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
            for t in range(len(test)):
                  pred = m.predict(test[t])[0]
                  if pred in majority[t]:
                        majority[t][pred] += 1
                  else:
                        majority[t][pred] = 1
                  if pred == actual[t]:
                        if pred == 1:
                              tmp['tp'] += 1
                        else:
                              tmp['tn'] += 1
                  else:
                        if pred == 1:
                              tmp['fp'] += 1
                        else:
                              tmp['fn'] += 1

            results[m.__class__.__name__].append(tmp)

      tmp = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
      for t in range(len(test)):
            maxNum = -1
            pred = -1
            for r in majority[t].keys():
                  if majority[t][r] > maxNum:
                        pred = r
                        maxNum = majority[t][r]
            if pred == actual[t]:
                  if pred == 1:
                        tmp['tp'] += 1
                  else:
                        tmp['tn'] += 1
            else:
                  if pred == 1:
                        tmp['fp'] += 1
                  else:
                        tmp['fn'] += 1
      results['majority'].append(tmp)

      for m in methods:
            print(m)
            #print(avg(results[m.__class__.__name__]))
            print(results[m.__class__.__name__])
            print("\n")
      print(results['majority'])
      """
      tmp = {'t':0, ':(':0}
      for t in range(len(test)):
            pred = mlc.classify_image(test[t])
            if pred == actual[t]:
                  tmp['t'] += 1
            else:
                  tmp[':('] += 1
      results['mlc'].append(tmp)

            #return info, svm, nb, knn, randFor, ada, neuralnet
def undersampling(biased, answers, times=1):
      tmp, tmp1, tmp2, first, second, ans = [], [], [], [], [], []
      for k in range(len(biased)):
            if answers[k] == 0:
                  tmp1.append(biased[k])
                  first.append(k)
            else:
                  tmp2.append(biased[k])
                  second.append(k)
      shuffle(tmp1)
      shuffle(tmp2)
      for k in range(len(second)):
            tmp.append(tmp1[k])
            ans.append(answers[first[k]])
            tmp.append(tmp2[k])
            ans.append(answers[second[k]])
      for k in range(len(second)*(times-1)):
            tmp.append(tmp1[k+len(second)])
            ans.append(answers[first[k+len(second)]])
      return tmp, ans

def bigClassifiation1(feats, classes):
    kf = KFold(n_splits=splits, shuffle=True)
    lassocv = LassoCV()
    svm = SVC()
    nb = GaussianNB()
    knn = KNeighborsClassifier()
    #randFor = RandomForestClassifier()
    #ada = AdaBoostClassifier()
    #neuralnet = MLPClassifier()

    #methods = [svm, nb, knn, randFor, ada, lassocv, neuralnet]
    methods = [lassocv, nb, svm, knn]
    results = {m.__class__.__name__:[] for m in methods}
    #results['majority'] = []


    for train_index, test_index in kf.split(feats):
        training, tests = [feats[i] for i in train_index], [feats[j] for j in test_index]
        classes, actuals = [classes[i] for i in train_index], [classes[j] for j in test_index]


        # LASSO Regression
        print("lasso")
        lassocv.fit(training, classes)

        #Support Vector Machines
        print("svm")
        svm.fit(training, classes)

        # Naive Bayes
        print("nb")
        nb.fit(training, classes)

        #K-Nearest Neighbors
        print("knn")
        knn.fit(training, classes)

        # Random Forest
        #print("rand for")
        #randFor.fit(training, classes)

        # AdaBoost
        #print("ada")
        #ada.fit(training, classes)

        # Neural Net (Backward Propogation)
        #print("neuralnet")
        #neuralnet.fit(training, classes)

        for m in methods:
            tmp = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
            for t in range(len(tests)):
                pred = m.predict(tests[t])
                if pred == actuals[t]:
                    if pred == 1:
                        tmp['tp'] += 1
                    else:
                        tmp['tn'] += 1
                else:
                    if pred == 1:
                        tmp['fp'] += 1
                    else:
                        tmp['fn'] += 1
            results[m.__class__.__name__].append(tmp)


    with open('./results/classificationsAnamoly.txt', 'w') as cout:
        for m in methods:
            cout.write(m)
            cout.write("\n")
            #print(avg(results[m.__class__.__name__]))
            cout.write(results[m.__class__.__name__])
            cout.write("\n")


def classificationBig(fpath, d, times, splits=10):
    kf = KFold(n_splits=splits, shuffle=True)
    XX, yy = readInput(fpath+d+".txt")

    #XX, yy = readInputWithID(fpath+d+".txt")
    lassocv = LassoCV()
    svm = SVC()
    nb = GaussianNB()
    #knn = KNeighborsClassifier()
    #randFor = RandomForestClassifier()
    #ada = AdaBoostClassifier()
    #neuralnet = MLPClassifier()

    #methods = [svm, nb, knn, randFor, ada, lassocv, neuralnet]
    methods = [lassocv, nb, svm]
    results = {m.__class__.__name__:[] for m in methods}


    for train_index, test_index in kf.split(XX):
        training, tests = [XX[i] for i in train_index], [XX[j] for j in test_index]
        classes, actuals = [yy[i] for i in train_index], [yy[j] for j in test_index]

        training, classes = undersampling(training, classes, times)


        # LASSO Regression
        print("lasso")
        lassocv.fit(training, classes)

        #Support Vector Machines
        print("svm")
        svm.fit(training, classes)

        # Naive Bayes
        print("nb")
        nb.fit(training, classes)

        #K-Nearest Neighbors
        print("knn")
        knn.fit(training, classes)

        # Random Forest
        #print("rand for")
        #randFor.fit(training, classes)

        # AdaBoost
        #print("ada")
        #ada.fit(training, classes)

        # Neural Net (Backward Propogation)
        #print("neuralnet")
        #neuralnet.fit(training, classes)

        for m in methods:
            tmp = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
            for t in range(len(tests)):
                pred = m.predict(tests[t])
                if pred == actuals[t]:
                    if pred == 1:
                        tmp['tp'] += 1
                    else:
                        tmp['tn'] += 1
                else:
                    if pred == 1:
                        tmp['fp'] += 1
                    else:
                        tmp['fn'] += 1
            results[m.__class__.__name__].append(tmp)

    with open('./results/classificationsBasic-' + str(index) + '.txt', 'w') as cout:
        for m in methods:
            cout.write(m)
            cout.write("\n")
            #print(avg(results[m.__class__.__name__]))
            cout.write(results[m.__class__.__name__])
            cout.write("\n")


def classification(training, classes, tests, actuals):
      # LASSO Regression
      lassocv = LassoCV().fit(training, classes)

      #Support Vector Machines
      svm = SVC()
      svm.fit(training, classes)

      # Naive Bayes
      nb = GaussianNB()
      nb.fit(training, classes)

      #K-Nearest Neighbors
      knn = KNeighborsClassifier()
      knn.fit(training, classes)

      # Random Forest
      randFor = RandomForestClassifier()
      randFor.fit(training, classes)

      # AdaBoost
      ada = AdaBoostClassifier()
      ada.fit(training, classes)

      # Neural Net (Backward Propogation)
      neuralnet = MLPClassifier()
      neuralnet.fit(training, classes)

      methods = [svm, nb, knn, randFor, ada, neuralnet, lassocv]
      results = {m.__class__.__name__:[] for m in methods}
      #index = {p: l for l,p in enumerate(powerset(results.keys())) if p != ''}
      #info = []
      #print([k for k in index.keys() if index[k] == 27])

      for m in methods:
            results[m.__class__.__name__] = m.score(tests, actuals)
            print(m)
            print(results[m.__class__.__name__])

      #return info, svm, nb, knn, randFor, ada, neuralnet

def masterTest(tests, actuals, dataset, **methods):
      training = []
      classification = []
      statistics = []

      for k in methods['data']:
            training.append(k[:-1])
            classification.append(k[-1])

      # import decoder
      m = ['GaussianNB', 'MLPClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'SVC', 'KNeighborsClassifier']
      index = {p: l for l,p in enumerate(powerset(m)) if p != ''}
      #print([k for k in index.keys() if index[k] == 27])

      # make list that includes NB
      nbMaster = GaussianNB()
      nbMaster.fit(training, classification)

      # train a NB (or something) on who does the best

      outfile = open(dataset+"MasterOut.txt", 'w')
      outfileNB = open(dataset+"NB.txt", 'w')
      outfileKNN = open(dataset+"KNN.txt", 'w')
      outfileRF = open(dataset+"RandomForest.txt", 'w')
      outfileSVM = open(dataset+"SVM.txt", 'w')
      outfileAda = open(dataset+"AdaBoost.txt", 'w')
      outfileNe = open(dataset+"NeuralNetwork.txt", 'w')

      for k in range(len(tests)):
            t = tests[k]
            which = nbMaster.predict(t)[0]
            meth = [j for j in index.keys() if index[j] == which][0]

            predNB = methods['nb'].predict(t)[0]
            predKNN = methods['knn'].predict(t)[0]
            predRand = methods['randFor'].predict(t)[0]
            predSVM = methods['svm'].predict(t)[0]
            predAda = methods['ada'].predict(t)[0]
            predNeur = methods['neuralnet'].predict(t)[0]


            """ # For game time
            if 'NB' in meth:
                  predicted = methods['nb'].predict(t)
            elif 'KNeighborsClassifier' in meth:
                  predicted = methods['knn'].predict(t)
            elif 'RandomForest' in meth:
                  predicted = methods['randomFor'].predict(t)
            elif 'SVC' in meth:
                  predicted = methods['svm'].predict(t)
            elif 'AdaBoost' in meth:
                  predicted = methods['ada'].predict(t)
            else:
                  predicted = methods['neuralnet'].predict(t)
            """

            if 'NB' in meth:
                  predicted = predNB
            elif 'KNeighborsClassifier' in meth:
                  predicted = predKNN
            elif 'RandomForest' in meth:
                  predicted = predRand
            elif 'SVC' in meth:
                  predicted = predSVM
            elif 'AdaBoost' in meth:
                  predicted = predAda
            else:
                  predicted = predNeur
            predicted = int(predicted)
            actuals[k] = int(actuals[k])

            """
            outfile.write(str(predicted)+","+str(actuals[k])+"\n")
            outfileNB.write(str(predNB)+","+str(actuals[k])+"\n")
            outfileNB.write(str(predKNN)+","+str(actuals[k])+"\n")
            outfileNB.write(str(predRand)+","+str(actuals[k])+"\n")
            outfileNB.write(str(predSVM)+","+str(actuals[k])+"\n")
            outfileNB.write(str(predAda)+","+str(actuals[k])+"\n")
            outfileNB.write(str(predNeur)+","+str(actuals[k])+"\n")
            """

            outfile.write(str(predicted)+","+str(actuals[k])+"\n")
            outfileNB.write(str(predNB)+","+str(actuals[k])+"\n")
            outfileKNN.write(str(predKNN)+","+str(actuals[k])+"\n")
            outfileRF.write(str(predRand)+","+str(actuals[k])+"\n")
            outfileSVM.write(str(predSVM)+","+str(actuals[k])+"\n")
            outfileAda.write(str(predAda)+","+str(actuals[k])+"\n")
            outfileNe.write(str(predNeur)+","+str(actuals[k])+"\n")

      outfile.close()
      outfileNB.close()
      outfileKNN.close()
      outfileRF.close()
      outfileSVM.close()
      outfileAda.close()
      outfileNe.close()


def readInput(file):
      #read csv
      with open(file) as temp_file:
            temp_file.readline()
            data = []
            temp_file.readline() # this shouldn't be needed, but an extra blank line was added at beginning
            for cin in temp_file:
                  try:
                        tmp = [float(r) for r in cin.strip('\n').strip().split(',')]
                        data.append(tmp)
                  except:
                        tmp = [r for r in cin.strip("\n").strip().split(',')]
                        if tmp[-1] == 'no':
                              tmp[-1] = 0
                        if isinstance(tmp[-1], str):
                              if tmp[-1].strip() == 'yes':
                                    tmp[-1] = 1
                        data.append([float(r) for r in tmp])

            #data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]

      #split data into
      t1, t2=[], []

      maxs = [-100 for _ in data[0]]
      for k in data:
            for j in range(len(k)):
                  if k[j] > maxs[j]:
                        maxs[j] = deepcopy(k[j])

      for k in range(len(data)):
            for j in range(len(data[k])):
                  if isnan(data[k][j]):
                        data[k][j] = maxs[j]*10
            t1.append(data[k][:-1])
            t2.append(data[k][-1])

      return t1, t2

"""
def readInputWithID(file):
      #read csv
      with open(file) as temp_file:
            temp_file.readline()
            data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]
      #split data into
      t1, t2=[], []

      maxs = [-100 for _ in data[0]]
      for k in data:
            for j in range(len(k)):
                  if k[j] > maxs[j]:
                        maxs[j] = deepcopy(k[j])

      for k in range(len(data)):
            for j in range(len(data[k])):
                  if isnan(data[k][j]):
                        data[k][j] = maxs[j]*10
            t1.append(data[k][:-1])
            t2.append(data[k][-1])

      return t1, t2
"""

def readOneIn(file):
      #read csv
      with open(file) as temp_file:
            head = temp_file.readline().strip().split(',')
            tmp = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]

      # find columns which are the confidence level in another variable
      confidence = []
      for a in range(len(head)):
            for b in range(len(head)):
                  if head[a] in head[b] and head[a] != head[b]:
                        confidence.append([a,b])

      dats = []


      for t in tmp:
            dats.append(t[1:])

      meta, data, headers = metadataCreation(dats, head[1:])

      return data, meta, headers, confidence


"""
def readH5(file=None):
      a = h5py.File("./DataMining/Project/musicFiles/TRAAAAW128F429D538.h5")
      print(a)
"""

def readFormattedInput(file):
      #read csv
      with open(file) as temp_file:
            temp_file.readline()
            data = [[r for r in line.rstrip('\n').split(',')] for line in temp_file]

      print(data[0])
      data2 = [[] for _ in range(len(data[0]))]

      with open('./musicFiles/formattedMusic.txt', 'w') as cout:
            i = 0
            for row in data:
                  for l in range(len(row)):
                        if row[l] == 'nan':
                              row[l] = str(10)
                  cout.write(','.join(row)+"\n")

      """
      data2.append(','.join([str(i)]+row[4:]))
      i += 1

      with open('./musicFiles/secondMusic.txt', 'w') as cout:
            for row in data2:
                  cout.write(row+"\n")
      """
      exit()

def readInputComplicated(file):

      #read csv
      with open(file) as temp_file:
            temp_file.readline()
            data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]
      #split data into
      t1, t2, t3, t4 =[], [], [], []
      tmp1 = data[len(data)//5:]
      tmp2 = data[:len(data)//5]

      for k in tmp1:
            t1.append(k[:-1])
            t2.append(k[-1])
      for j in tmp2:
            t3.append(j[:-1])
            t4.append(j[-1])

      return t1, t2, t3, t4


def doAnalysis(file):
      with open(file) as temp_file:
            data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]

def metadataCreation(data, hdr):
      tmpData = [[] for _ in range(len(data[0]))]

      for l in data:
            for i in range(len(data[0])):
                  tmpData[i].append(l[i])

      i = 0
      outs = []

      for l in tmpData:
            distinct = []
            for k in l:
                  if k not in distinct:
                        distinct.append(k)
            if len(distinct) == 1:
                  outs.append(i)
            i += 1

      outheaders = [j for i,j in enumerate(hdr) if i not in outs]

      if len(outs) > 0:
            tmpdata = deepcopy(data)
            data = []
            data = [[j for i, j in enumerate(row) if i not in outs] for row in tmpdata]

      # Estimate which variables are continuous
      tmpData = [[] for _ in range(len(data[0]))]
      cts = [True for _ in range(len(data[0]))]

      for l in data:
            for i in range(len(data[0])):
                  tmpData[i].append(l[i])

      i = 0
      for l in tmpData:
            distinct = []
            for k in l:
                  if k not in distinct:
                        distinct.append(k)
            if len(distinct) < len(data)//25:
                  cts[i] = False

            i += 1

      return cts, data, outheaders

"""
with open('./musicFiles/millionSongString.txt', 'w') as cout:
      with open('./musicFiles/millionSongs.csv',  'r') as cin:
            cout.write(cin.readline())
            cout.write("\n")
            #tmp = [[r.strip().split(',')[0]] + r.strip().split(',')[4:] for r in cin]
            tp = [r.strip().split(',') for r in cin]
            tmp = []
            tmp1 = []
            tmp2 = []
            for r in tp:
                  tmp1.append(r[:4])
                  tmp2.append(r[4:])
            for t in range(len(tmp1)):
                  if tmp2[t][-1] != 0 and tmp2[t][-1] != 1:
                        if tmp2[t][-1] == 'no':
                              tmp2[t][-1] = 0
                        elif tmp2[t][-1] == 'yes':
                              tmp2[t][-1] == 1
                        else:
                              print("Ugh")
                              print(tmp2[t][-1])
                  tmp.append([hash(t2) for t2 in tmp1[t]] + tmp2[t])
                  cout.write(','.join([str(r) for r in [hash(t2) for t2 in tmp1[t]] + tmp2[t]]))
                  cout.write("\n")

exit()
"""

fpath = './'
dataFiles = ['millionSongs3']
file = 'millionSongString'
#outlierDetection(fpath, file)
outlierFeats, outlierClasses = outlierPreprocessing(fpath, file)
bigClassifiation1(outlierFeats, outlierClasses)
for kk in [1,3,5]:
    classificationBig(fpath, file, kk)
exit()

fpath = './musicFiles/'
dataFiles = ['formattedMusic']

for d in dataFiles:
      XX, yy = readInput(fpath+d+".txt")
      for train_index, test_index in kf.split(XX):
            X_train, X_test = [XX[i] for i in train_index], [XX[j] for j in test_index]
            y_train, y_test = [yy[i] for i in train_index], [yy[j] for j in test_index]
            classification(X_train, y_train, X_test, y_test)
            exit()


fpath = './musicFiles/'
dataFiles = ['finalData3']
for d in dataFiles:
      X, meta, headers, conf = readOneIn(fpath+d+'.txt')
      #clustering(X)
      #print(meta)
      #print(len(X[0]))
      projectClassification(X, meta, headers, conf)
exit()
for d in dataFiles:
      X, y = readInput(fpath+d+".txt")
      Z = deepcopy(X)
      for entry in range(len(y)):
            Z[entry].append(y[entry])
      outlierDetection(Z,y)

kf = KFold(n_splits=4, shuffle=True)
for d in dataFiles:
      #X, y = readInput(fpath+d+".txt")
      X, y, XX, yy = readInputComplicated(fpath+d+".txt")
      #with open(fpath+d+"out.txt", 'w') as cout:
      dat, sv, nbs, kn, ran, ad, neu = classification(X[len(X)//5:], y[len(y)//5:], X[:len(X)//5], y[:len(y)//5])
      #with open(fpath+"/"+d+"/MasterOut.txt", 'w') as cout:
      for train_index, test_index in kf.split(XX):
            X_train, X_test = [XX[i] for i in train_index], [XX[j] for j in test_index]
            y_train, y_test = [yy[i] for i in train_index], [yy[j] for j in test_index]
            masterTest(X_test, y_test, fpath+"/"+d+"/", data=dat, svm=sv, nb=nbs, knn=kn, randFor=ran, ada=ad, neuralnet=neu)


"""
for d in dataFiles:
      doAnalysis(fpath+d+"out.txt")
"""



"""
from email.mime.text import MIMEText
def send_email():
      # Create the enclosing (outer) message
      outer = MIMEMultipart()
      outer['Subject'] = 'Progress Report %s' % os.path.abspath(directory) # This needs to be something that says which file called it.
      outer['To'] = 'ahill6@ncsu.edu'
      outer['From'] = opts.sender # need to fix this

      for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if not os.path.isfile(path):
                  continue

            fp = open(path)
            # Note: we should handle calculating the charset
            msg = MIMEText(fp.read(), _subtype=subtype)
            fp.close()

            # Set the filename parameter
            msg.add_header('Content-Disposition', 'attachment', filename=filename)
            outer.attach(msg)
            # Now send or store the message
            composed = outer.as_string()
            s = smtplib.SMTP('localhost')
            s.sendmail(sender, recipients, composed)
            s.quit()
"""
# read in file
# change 'no' to 0, 'yes' to 1
# write file
