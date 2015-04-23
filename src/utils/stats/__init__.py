import numpy as np
import functools
import math,os
import numpy as np

foldl = functools.reduce
NBSTATS = 5
(TP, FP, FN, TN, NBSHADOWERROR) = range(NBSTATS)

def comparator(gt_mask,ac_mask,roi_mask=None):
	if (np.sum(gt_mask==1)==0):
		return [0,0,0,0,0]
	if roi_mask is None:
		roi_mask = np.ones(gt_mask.shape,dtype=np.uint8);
	tp = np.sum((gt_mask==1) & (ac_mask==1) & (roi_mask==1));
	fp = np.sum((gt_mask==0) & (ac_mask==1) & (roi_mask==1));
	fn = np.sum((gt_mask==1) & (ac_mask==0) & (roi_mask==1));
	tn = np.sum((gt_mask==0) & (ac_mask==0) & (roi_mask==1));
	return [tp,fp,fn,tn,0]

def updateConfusion(accumulator,confusion):
	for idx in range(5):
		accumulator[idx] += confusion[idx];

def getNewConfusion():
	return [0.0]*5;

def addVectors(lst1, lst2):
    """Add two lists together like a vector."""
    return list(map(float.__add__, lst1, lst2))

def sumListVectors(lst):
    """Calculate the sum of a list of 4-values vector."""
    return foldl(addVectors, lst, [0.0, 0.0, 0.0, 0.0, 0.0])

def getStats(cm):
    """Return the usual stats for a confusion matrix."""
    TP, FP, FN, TN, SE = cm
    recall = TP / (TP + FN + 0.00000001)
    specficity = TN / (TN + FP + 0.00000001)
    fpr = FP / (FP + TN + 0.00000001)
    fnr = FN / (TP + FN + 0.00000001)
    pbc = 100.0 * (FN + FP) / (TP + FP + FN + TN + 0.00000001)
    precision = TP / (TP + FP + 0.00000001)
    fmeasure = 2.0 * (recall * precision) / (recall + precision + 0.00000001)
    return [recall, specficity, fpr, fnr, pbc, precision, fmeasure]

def mean(l):
    """Return the mean of a list."""
    return sum(l) / len(l)

def cmToText(cm):
    return ' '.join([str(val) for val in cm])

def writeComment(f):
    f.write('#This is the statistical file we use to compare each method.\n')
    f.write('#Only the lines starting with "cm" are importants.\n')
    f.write('#cm NbTruePositive NbFalsePositive NbFalseNegative NbTrueNegative NbErrorShadow\n\n')

class Stats:
    def __init__(self, path):
        self.path = path
        self.categories = dict()

    def addCategories(self, category):
        if category not in self.categories:
            self.categories[category] = {}

    def update(self, category, video, confusionMatrix):
        self.categories[category][video] = confusionMatrix

    def writeCategoryResult(self, category):
        """Write the result for each category."""
        videoList = list(self.categories[category].values())
        categoryStats = []
        categoryTotal = sumListVectors(videoList)
        filepath  = os.path.join(self.path,str(category),'stats.txt')
        with open(filepath, 'w') as f:
            writeComment(f)
            for video, cm in self.categories[category].items():
                categoryStats.append(getStats(cm))
                f.write('cm video ' + category + ' ' + video + ' ' + cmToText(cm) + '\n')
                
            f.write('cm category ' + category + ' ' + cmToText(categoryTotal) + '\n\n')
            f.write('\nRecall\t\t\tSpecificity\t\tFPR\t\t\t\tFNR\t\t\t\tPBC\t\t\t\tPrecision\t\tFMeasure')
            f.write('\n{0:1.10f}\t{1:1.10f}\t{2:1.10f}\t{3:1.10f}\t{4:1.10f}\t{5:1.10f}\t{6:1.10f}'.format(*[mean(z) for z in zip(*categoryStats)]))
        
    def writeOverallResults(self):
        """Write overall results."""
        totalPerCategoy = [sumListVectors(list(CMs.values())) for CMs in self.categories.values()]
        categoryStats = {}
        
        with open(self.path + os.sep + 'stats.txt', 'w') as f:
            writeComment(f)
            for category in self.categories.keys():
                videoList = list(self.categories[category].values())
                categoryStats[category] = []
                categoryTotal = sumListVectors(videoList)
                for video, cm in self.categories[category].items():
                    categoryStats[category].append(getStats(cm))
                    f.write('stat  ' + category + ' ' + video + ' ' + cmToText(cm) + '\n')
                f.write('stat category ' + category + ' ' + cmToText(categoryTotal) + '\n\n')
                cur = [mean(z) for z in zip(*categoryStats[category])]
                categoryStats[category] = cur
            total = sumListVectors(totalPerCategoy)
            f.write('\n\ncm overall ' + cmToText(total))

            overallStats = []
            f.write('\n\n\n\n\t\t\tRecall\t\t\tSpecificity\t\tFPR\t\t\t\tFNR\t\t\t\tPBC\t\t\t\tPrecision\t\tFMeasure')
            for category, stats in categoryStats.items():
                overallStats.append(stats)
                if len(category) > 8:
                    category = category[:7]
                f.write('\n{0} :\t{1:1.10f}\t{2:1.10f}\t{3:1.10f}\t{4:1.10f}\t{5:1.10f}\t{6:1.10f}\t{7:1.10f}'.format(category, *stats))

            f.write('\n\nOverall :\t{0:1.10f}\t{1:1.10f}\t{2:1.10f}\t{3:1.10f}\t{4:1.10f}\t{5:1.10f}\t{6:1.10f}'.format(*[mean(z) for z in zip(*overallStats)]))
            if len(categoryStats) < 6:
                f.write('\nYour method will not be visible in the Overall section.\nYou need all 6 categories to appear in the overall section.')
