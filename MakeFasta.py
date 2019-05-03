#http://castor.bioinfo.uqam.ca/index.php?castor=build

import pickle

train = pickle.load(open("C:/Users/an_fab/Desktop/train4.p", "rb"))

labels = open("C:/Users/an_fab/Desktop/labels.txt", "w")
genomes = open("C:/Users/an_fab/Desktop/genomes.txt", "w")

for i in range(len(train)):

  idx = 'HIV%04d' % (i,)
  labels.write(idx)
  labels.write(",")
  labels.write(train[i][0])
  labels.write("\n")

  genomes.write(">")
  idx = 'HIV%04d %s' % (i,train[i][0],)
  genomes.write(idx)
  genomes.write("\n")
  genomes.write(train[i][1].strip())
  genomes.write("\n")

labels.close()
genomes.close()

test = pickle.load(open("C:/Users/an_fab/Desktop/test4.p", "rb"))

labels = open("C:/Users/an_fab/Desktop/test_labels.txt", "w")
genomes = open("C:/Users/an_fab/Desktop/test_genomes.txt", "w")

for i in range(len(test)):
      
  idx = 'HIV.TEST%04d' % (i,)
  labels.write(idx)
  labels.write(",")
  labels.write(test[i][0])
  labels.write("\n")

  genomes.write(">")
  idx = 'HIV.TEST%04d %s' % (i,test[i][0],)
  genomes.write(idx)
  genomes.write("\n")
  genomes.write(test[i][1].strip())
  genomes.write("\n")

labels.close()
genomes.close()
