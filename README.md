# car_prediction
predict different features of given csv using different classifiers 
It is a sample program to verify the prediction result of differnt classifers
# Aim of this article
We will use different multiclass classification methods such as, KNN, Decision trees, SVM, etc. We will compare their accuracy on test data.
# Decision tree classifier 
Decision tree classifier is a systematic approach for multiclass classification. It poses a set of questions to the dataset (related to its attributes/features). The decision tree classification algorithm can be visualized on a binary tree. On the root and each of the internal nodes, a question is posed and the data on that node is further split into separate records that have different characteristics. The leaves of the tree refer to the classes in which the dataset is split. In the following code snippet, we train a decision tree classifier in scikit-learn
# SVM (Support vector machine) classifier 
SVM (Support vector machine) is an efficient classification method when the feature vector is high dimensional. In sci-kit learn, we can specify the the kernel function (here, linear). To know more about kernel functions and SVM refer – Kernel function | sci-kit learn and SVM.
# KNN (k-nearest neighbours) classifier 
KNN or k-nearest neighbours is the simplest classification algorithm. This classification algorithm does not depend on the structure of the data. Whenever a new example is encountered, its k nearest neighbours from the training data are examined. Distance between two examples can be the euclidean distance between their feature vectors.
# Naive Bayes classifier
Naive Bayes classification method is based on Bayes’ theorem. It is termed as ‘Naive’ because it assumes independence between every pair of feature in the data. Let (x1, x2, …, xn) be a feature vector and y be the class label corresponding to this feature vector.
# References –

    http://scikit-learn.org/stable/modules/naive_bayes.html
    https://en.wikipedia.org/wiki/Multiclass_classification
    http://scikit-learn.org/stable/documentation.html
    http://scikit-learn.org/stable/modules/tree.html
    http://scikit-learn.org/stable/modules/svm.html#svm-kernels
    https://www.analyticsvidhya.com/blog/2015/10/understaing-support-vector-machine-example-code/
