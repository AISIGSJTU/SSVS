import os
import sys
import glob
import cv2
from pathlib import Path
import numpy
from random import shuffle
from tqdm import trange
import sys


#metrics
def specificity(result, reference):
    """
    Specificity.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    specificity : float
        The specificity between two binary datasets, here mostly binary objects in images,
        which denotes the fraction of correctly returned negatives. The
        specificity is not symmetric.

    See also
    --------
    :func:`sensitivity`

    Notes
    -----
    Not symmetric. The completment of the specificity is :func:`sensitivity`.
    High recall means that an algorithm returned most of the irrelevant results.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))

    tn = numpy.count_nonzero(~result & ~reference)
    fp = numpy.count_nonzero(result & ~reference)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity

def dc(result, reference):
    r"""
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))

    intersection = numpy.count_nonzero(result & reference)

    size_i1 = numpy.count_nonzero(result)
    size_i2 = numpy.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc

def jc(result, reference):
    """
    Jaccard coefficient

    Computes the Jaccard coefficient between the binary objects in two images.

    Parameters
    ----------
    result: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    reference: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `result` and the
        object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))

    intersection = numpy.count_nonzero(result & reference)
    union = numpy.count_nonzero(result | reference)

    jc = float(intersection) / float(union)

    return jc

def precision(result, reference):
    """
    Precison.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    precision : float
        The precision between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of retrieved instances that are relevant. The
        precision is not symmetric.

    See also
    --------
    :func:`recall`

    Notes
    -----
    Not symmetric. The inverse of the precision is :func:`recall`.
    High precision means that an algorithm returned substantially more relevant results than irrelevant.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))

    tp = numpy.count_nonzero(result & reference)
    fp = numpy.count_nonzero(result & ~reference)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision

def recall(result, reference):
    """
    Recall.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    recall : float
        The recall between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of relevant instances that are retrieved. The
        recall is not symmetric.

    See also
    --------
    :func:`precision`

    Notes
    -----
    Not symmetric. The inverse of the recall is :func:`precision`.
    High recall means that an algorithm returned most of the relevant results.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))

    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall



def main(path='pred'):

  filepath_list = list(Path(path).glob('*'))
  result_precision = []
  result_recall = []
  result_jaccard = []
  result_dice = []
  result_sp = []

  #with open('/home/lumos/CAD/data/log.txt','a')as f:
       #f.writelines('Precision-------Recall----------Jaccard---------Dice------------IMG---------\n')
  for pred_path in filepath_list:
               gt_path = str(pred_path).replace(path,'masks')

               y_pred = cv2.imread(str(pred_path),0)
               y_true = cv2.imread(gt_path,0)
               
               p=precision(y_pred,y_true)
               r=recall(y_pred,y_true)
               j=jc(y_pred,y_true)
               d=dc(y_pred,y_true)
               s=specificity(y_pred,y_true)
               '''
               p=precision(y_true,y_pred)
               r=recall(y_true,y_pred)
               j=jc(y_true,y_pred)
               d=dc(y_true,y_pred) 
               '''
               result_precision += [p]
               result_recall += [r]
               result_jaccard += [j]
               result_dice += [d]
               result_sp += [s]
               '''
               with open('/home/lumos/CAD/data/log.txt','a')as f:
                  f.writelines(str(p)[0:9]+'\t'+str(r)[0:9]+'\t'+ str(j)[0:9]+'\t'+ str(d)[0:9]+'\t'+str(gt_path)[5:]+'\n')
               '''

  with open('log.txt','a')as f:
       f.writelines(path)
       f.writelines('---------------------------------------------------------------\n')
       f.writelines('Precision = '+ str(numpy.mean(result_precision))+','+ str(numpy.std(result_precision))+'\n')
       f.writelines('Recall = '+ str(numpy.mean(result_recall))+','+ str(numpy.std(result_recall))+'\n')
       f.writelines('Jaccard = '+ str(numpy.mean(result_jaccard))+','+ str(numpy.std(result_jaccard))+'\n')
       f.writelines('Dice = '+ str(numpy.mean(result_dice))+','+ str(numpy.std(result_dice))+'\n')
       f.writelines('Specificity = '+ str(numpy.mean(result_sp))+','+ str(numpy.std(result_sp))+'\n')
       f.writelines('\n')


if __name__ == '__main__':
    path=sys.argv[1]
    main(path)
