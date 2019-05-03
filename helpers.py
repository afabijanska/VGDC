import matplotlib.pyplot as plt
import numpy as np

#extract classess from a list of tuples class-genome

def getClasses(li):
    classes = set()
    
    for i in range(len(li)):
        classId = li[i][0]
        classes.add(classId)
    
    return classes

#get class occurence stats

def getStats(li):
    classes = getClasses(li)
    diClasses = {}

    for item in classes:
        diClasses[item] = 0

    tot = 0
    maxLen = 0;
    for i in range(len(li)):
        diClasses[li[i][0]] += 1
        tot += 1
        
        if len(li[i][1]) > maxLen:
            maxLen = len(li[i][1])

    print("-----------some stats: ------------")
    print("total num of classes: ", len(diClasses))
    print("total num of samples: ", tot)
    print("max genome length: ", maxLen)
    print("occurences: ")
    for key, item in diClasses.items():
       print(f'{key:8} => {item:8}')
    print("-----------------------------------")
    
    return diClasses

#plot histogram of class occurences

def plotDict(d, path):
    plt.bar(range(len(d)), d.values(), align='center')
    plt.xticks(range(len(d)), list(d.keys()))
    plt.xticks(rotation=90)
    plt.savefig(path) 
    plt.show()
    
#convert letters to numbers

def genome2tabInt(genome, maxLen):
    
    tabInt = np.zeros(maxLen)
    
    for i in range(len(genome)-1):
        if ord(genome[i]) != 32:
            tabInt[i] = ord(genome[i])
        
    return tabInt

#plot confusion matrix

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
