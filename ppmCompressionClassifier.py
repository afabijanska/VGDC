# Python 3.6+

import pickle, time, os, collections
import subprocess

FOLDS = 5   # filenames: train1.p / test1.p, ..., train5.p / test5.p


def makeGenomeDict(trainData):
  import collections
  di = collections.defaultdict(list)
  classes = set()
  for className, seq in trainData:
    classes.add(className)
    di[className] += [seq]
  di2 = {}
  for cl in classes:
    di2[cl] = "".join(di[cl])
  return di2, classes


def main():
  global FOLDS
  
  timeStart = time.time()

  correct = incorrect = 0
  for i in range(1, FOLDS + 1):
    print("Fold #{} starts...".format(i))
    trainData = pickle.load(open("./train" + str(i) + ".p", "rb"))
    assert type(trainData) == list
    trainDict, classes = makeGenomeDict(trainData)
    print("trainDict created")
    
    classToInt = {}
    for j, cl in enumerate(classes):
      classToInt[cl] = j
    
    testData = pickle.load(open("./test" + str(i) + ".p", "rb"))
    assert type(testData)  == list
    
    compressionResults = collections.defaultdict(list)
    
    for j, cl in enumerate(classes):
      currClassFileName = str(j).zfill(4) + "_class"
      print(type(trainDict[cl]))
      open(currClassFileName, "wb").write(trainDict[cl].encode("utf-8"))
      # PPMTrain.exe -mXX -oYY -sZZ SomeTrainingFile
      subprocess.call(["PPMTrain.exe", "-m32", "-o8", currClassFileName])
      
      for jj, (className, testSeq) in enumerate(testData):
        testSeqFileName = str(classToInt[className]).zfill(4) + "_" + str(jj).zfill(4)
        open(testSeqFileName, "wb").write(testSeq.encode("utf-8"))
        # PPMd1 e -m32 -o8 SomeRealFile
        subprocess.call(["PPMd1.exe", "e", "-m32", "-o8", testSeqFileName])
        size_ = os.path.getsize(testSeqFileName + ".pmd")
        compressionResults[ jj ] += [(classToInt[className], j, size_)]
        os.remove(testSeqFileName)
        os.remove(testSeqFileName + ".pmd")        
    
    for k, v in compressionResults.items():
      trueClass = v[0][0]
      min_ = min(v, key = lambda x: x[2])
      if min_[1] == trueClass:
        correct += 1
      else:
        incorrect += 1
      
  timeEnd = time.time()
  print("TOTAL RESULTS. Correct test samples: {}, incorrect test samples: {}".format(correct, incorrect))
  print("Accuracy = {:7.3f} %".format(100.0 * correct / (correct + incorrect)))
  print("Elapsed time: {} seconds".format(timeEnd - timeStart))


if __name__ == "__main__":
  main()
