import os
import pickle
import matplotlib
"""if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    print 'DISPLAY not in this enviroment'
    matplotlib.use('Agg')
"""
import matplotlib.pyplot as plt
import numpy as np

debug_flag=True
def plotROC(predStrength , labels):
    matplotlib.use('Agg')
    assert np.ndim(predStrength) == np.ndim(labels)
    if np.ndim(predStrength) ==2:
        predStrength=np.argmax(predStrength , axis=1)
        labels=np.argmax(labels, axis=1)



    #how to input?

    cursor=(1.0,1.0) #initial cursor
    ySum= 0.0 # for AUC curve
    n_pos=np.sum(np.array(labels) ==1)
    n_neg=len(labels)-n_pos
    y_step=1/float(n_pos)
    x_step=1/float(n_neg)
    n_est_pos=0
    sortedIndices=np.argsort(predStrength , axis=0)
    fig= plt.figure()
    fig.clf()
    ax=plt.subplot(1,1,1)
    if __debug__ == debug_flag:
        print 'labels',labels[:10]
        print 'predStrength',predStrength.T[:10]
        print 'sortedIndices',sortedIndices.T[:10]
        print  sortedIndices.tolist()[:10]
    for ind in sortedIndices.tolist():
        print ind
        if labels[ind] ==1.0:
            DelX=0; DelY=y_step
        else :
            DelX=x_step ; DelY=0
            ySum += cursor[1]
        ax.plot([ cursor[0] , cursor[0]-DelX ] , [ cursor[1] , cursor[1]-DelY])
        cursor=(cursor[0]-DelX , cursor[1] -DelY)
        if __debug__ == debug_flag:
            print 'label',labels[ind]
            print 'delX',
            print 'sortedIndices', sortedIndices.T
            print 'DelX:',DelX,'DelY:',DelY
            print 'cursor[0]-DelX :',cursor[0],'cursor[1]-DelY :',cursor[1]
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Fundus Classification System')
    ax.axis([0,1,0,1])
    if __debug__==debug_flag:
        print '# of True :' ,n_pos
        print '# of False :' ,n_neg
    plt.savefig('roc.png')
    #plt.show()
    print 'The Area Under Curve is :' , ySum*x_step

if '__main__' == __name__ :
    preds=np.load('best_preds.npy')
    labels = np.load('labels.npy')
    print preds
    plotROC(preds, labels)