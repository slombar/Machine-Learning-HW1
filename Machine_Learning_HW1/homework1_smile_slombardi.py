import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC(y, yhat):
    # comparison of labels
    c = np.equal(y, np.transpose(yhat))
    # total num of true predictions over total labels
    return np.count_nonzero(c, axis=1) / y.shape[0]

def measureAccuracyOfPredictors(predictors, X, y):
    # pixel value predictors for each image
    pixelVal1 = X[:, predictors[:, :, 0], predictors[:, :, 1]]
    pixelVal2 = X[:, predictors[:, :, 2], predictors[:, :, 3]]

    # compare all of the predictor pixels
    results = pixelVal1 > pixelVal2
    del pixelVal1
    del pixelVal2

    # calculate mean of all predictors
    avgResults = np.mean(results, axis=2)
    del results

    # convert from int to boolean (yes smiling no not smiling)
    results = np.greater(avgResults, np.full(avgResults.shape, .5))
    del avgResults

    return fPC(y, results)

def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, n):
    # predictors, batch size, feature size
    pred = np.full((6, 4), None)
    bs = 100
    m = 6

    # find all permutations for 0-23
    idxs = np.arange(0, 24)
    allI = np.array(np.meshgrid(idxs, idxs, idxs, idxs)).T.reshape(-1, 4)

    ''' 
    For each feature, 
    at each round j, choose the jth feature such that – when it is added to the set of j − 1 features that have already been selected – the
    accuracy (fPC) of the overall classifier on the training set is maximized.
    
    Run training and testing accuracy for m=6. 
    '''
    for i in range(m):
        # fill all predictors with -1
        #331776 is 24^4 (two 24x24 matricies)
        allPred = np.full((331776, i + 1, 4), -1)

        # add the previous predictor to the list of new predictors
        allPred[:, :i, :] = pred[:i, :]

        # all possible new predictors
        allPred[:, i, :] = allI

        '''
        Use batches to save computation time
        Measure the accuracy of the given predictors on the training faces 
        '''
        # batch number and accuracy
        bn = 0
        trainAcc = list()

        '''
        TRAIN ACCURACY
        
        While batch num is still smaller than n
        If n - batch num is greater than or equal to the batch size, we will append the measurement of this
        prediction to the accuracy list.
        If n - batch num is less than the batch size, we have reached the final element. Now we can run our accuracy
        measurement only on the last feature, and end the loop by setting bn = n.
        '''
        while (bn < n):

            if (n - bn >= bs):

                trainAcc.append(measureAccuracyOfPredictors(allPred, trainingFaces[bn:(bn + bs)], trainingLabels[bn:(bn + bs)]))
                bn += bs

            elif (n - bn < bs):
                trainAcc.append(measureAccuracyOfPredictors(allPred, trainingFaces[bn:], trainingLabels[bn:]))
                bn = n

        #average our scores to get our final (training data) accuracy score
        trainAcc = np.array(trainAcc)
        trainAcc = np.mean(trainAcc, axis=0)

        #locate our best predictor
        max = np.argmax(trainAcc)

        #save our best predictor
        pred[i, 0] = allPred[max, i, 0]
        pred[i, 1] = allPred[max, i, 1]
        pred[i, 2] = allPred[max, i, 2]
        pred[i, 3] = allPred[max, i, 3]

        '''
        TEST ACCURACY
        '''
        bn = 0
        trainAcc = list()
        while (bn < testingFaces.shape[0]):

            if (testingFaces.shape[0] - bn >= bs):

                trainAcc.append(measureAccuracyOfPredictors(np.array([allPred[max]]),
                                                           testingFaces[bn:(bn + bs)],
                                                           testingLabels[bn:(bn + bs)]))

                bn += bs

            elif (testingFaces.shape[0] - bn < bs):
                trainAcc.append(measureAccuracyOfPredictors(np.array([allPred[max]]),
                                                           testingFaces[bn:],
                                                           testingLabels[bn:]))

                bn = testingFaces.shape[0]

        trainAcc = np.array(trainAcc)
        trainAcc = np.mean(trainAcc, axis=0)

    return pred

#Load data from correct training/testing set and reshape faces to fit 24x24
def loadData(t):
    if(t):
        faces = np.load("data\\training\\trainingFaces.npy")
        labels = np.load("data\\training\\trainingLabels.npy")
    else:
        faces = np.load("data\\testing\\testingFaces.npy")
        labels = np.load("data\\testing\\testingLabels.npy")

    faces = faces.reshape(-1, 24, 24)
    return faces, labels




if __name__ == "__main__":
    testingFaces, testingLabels = loadData(0)
    trainingFaces, trainingLabels = loadData(1)

    # show image or nah
    show = True

    # analyze how training/testing accuracy changes as a function of number of examples
    '''
    400 
    600 
    800 
    1000 
    1200 
    1400 
    1600
    1800 
    2000
    '''
    n = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    p = None

    '''
    for each element in n, run stepwise regression on those examples from the training set
    Measure and record the training accuracy of the trained classifier on the n examples.
    Measure and record the testing accuracy of the classifier on the (entire) test set
    '''
    for i in n:
        p = stepwiseRegression(trainingFaces[:i], trainingLabels[:i], testingFaces, testingLabels, i)

    '''Draw image of face with red and blue patches'''
    if show:
        image = testingFaces[0, :, :]
        fig, ax = plt.subplots(1)
        #display first pixel, r1,c1
        ax.imshow(image, cmap='gray')

        for r1, c1, r2, c2 in p:
            rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            #Display 2nd pixel, r2,c2
            ax.add_patch(rect)
            #merge and display results
            rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

        #print photo
        plt.show()
