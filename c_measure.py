# Python 3.x

import pickle, random, time


FOLDS = 5   # filenames: train1.p / test1.p, ..., train5.p / test5.p

Kmers = [10, 20, 30, 40, 50, 60, 70, 80]


def makeGenomeDict(trainData, k):
  import collections
  di = collections.defaultdict(set)
  classes = set()
  for className, seq in trainData:
    classes.add(className)
    for i in range(len(seq) - k + 1):
      di[className].add(seq[i : i + k])
  return di, classes


def classify(testData, k, trainDict, classes):
  print("testData length = {}".format(len(testData)))
  correct = incorrect = 0
  counter = 0
  for className, seq in testData:
    counts = {cl : 0 for cl in classes}
    kmerSet = set()
    for i in range(len(seq) - k + 1):
      kmerSet.add(seq[i : i + k])
    for cl in classes:
      counts[cl] = len(kmerSet & trainDict[cl])
    outputClass = max(counts, key = lambda x: (counts[x], x))
    if className == outputClass:
      correct += 1
    else:
      incorrect += 1
    counter += 1
    if counter % 100 == 99: print(".", end = "", flush = True)
  print("\n")
  return correct, incorrect


def main():
  global Kmers
  global FOLDS
  
  trainTime = 0.0
  testTime = 0.0
  
  timeStart = time.time()

  correct = incorrect = 0
  
  testSamples = 0
  
  for i in range(1, FOLDS + 1):
    trainTimeStart = time.time()
    bestK = -1
    bestIncorrect = 999999999  
    # finding the best K
    for K in Kmers:  
      trainData = pickle.load(open("./train" + str(i) + ".p", "rb"))
      assert type(trainData) == list
      trainDataLen = len(trainData)
      random.shuffle(trainData)
      trainPart, validationPart = trainData[ : trainDataLen // 2], trainData[trainDataLen // 2 : ]
      trainDict, classes = makeGenomeDict(trainPart, K)
      corr, incorr = classify(validationPart, K, trainDict, classes)
      if incorr < bestIncorrect:
        bestK = K
        bestIncorrect = incorr
    print("Fold #{}, best K = {}".format(i, bestK))
    
    # training (again) acc. to the found K
    trainData = pickle.load(open("./train" + str(i) + ".p", "rb"))
    assert type(trainData) == list
    trainDict, classes = makeGenomeDict(trainData, bestK)
    print("trainDict created")
    trainTimeEnd = time.time()      
    trainTime += trainTimeEnd - trainTimeStart
    
    # classifying
    testTimeStart = time.time()
    testData = pickle.load(open("./test" + str(i) + ".p", "rb"))
    assert type(testData)  == list
    testSamples += len(testData)
        
    corr, incorr = classify(testData, bestK, trainDict, classes)
    correct += corr
    incorrect += incorr

    testTimeEnd = time.time()
    testTime += testTimeEnd - testTimeStart
    
    print("Fold #{}. Correct test samples: {}, incorrect test samples: {}\n".format(i, corr, incorr))

  timeEnd = time.time()
  print("TOTAL RESULTS. Correct test samples: {}, incorrect test samples: {}".format(correct, incorrect))
  print("Accuracy = {:7.3f} %".format(100.0 * correct / (correct + incorrect)))
  print("Total elapsed time: {} seconds".format(timeEnd - timeStart))
  print("Total training time: {} seconds".format(trainTime))
  print("Total testing (classification) time: {:10.3f} seconds".format(testTime))  
  print("Avg classification time per sample: {:10.3f} msec".format(1000.0 * testTime / testSamples))
  print("Classified samples per second: {:8.2f}".format(testSamples / testTime))


if __name__ == "__main__":
  main()
