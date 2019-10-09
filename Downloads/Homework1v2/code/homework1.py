from classifiers import *
from os import listdir
from matplotlib import image

# interpreting your performance with 100 training examples per category:
# accuracy  =   0 ->  your code is broken (probably not the classifier's
#                     fault! a classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .07 -> your performance is chance.
#  accuracy ~= .20 -> rough performance with tiny images and nearest
#                     neighbor classifier.
#  accuracy ~= .20 -> rough performance with tiny images and linear svm
#                     classifier. the linear classifiers will have a lot of
#                     trouble trying to separate the classes and may be
#                     unstable (e.g. everything classified to one category)
#  accuracy ~= .50 -> rough performance with bag of sift and nearest
#                     neighbor classifier.
#  accuracy ~= .60 -> you've gotten things roughly correct with bag of
#                     sift and a linear svm classifier.
#  accuracy >= .70 -> you've also tuned your parameters well. e.g. number
#                     of clusters, svm regularization, number of patches
#                     sampled when building vocabulary, size and step for
#                     dense sift features.
#  accuracy >= .80 -> you've added in spatial information somehow or you've
#                     added additional, complementary image features. this
#                     represents state of the art in lazebnik et al 2006.
#  accuracy >= .85 -> you've done extremely well. this is the state of the
#                     art in the 2010 sun database paper from fusing many 
#                     features. don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> you get to teach the class next year.
#  accuracy >= .96 -> you can beat a human at this task. this isn't a
#                     realistic number. some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.


if __name__ == "__main__":
    # KNN 
    # todo: add runtime using timeit

    # load all images in a directory
    X_train = list()
    y_train = list()
    for filename in listdir('../data/train'):
        for label in listdir('../data/train/' + filename):
            # print('../data/train/' + filename + '/' + label)
            # load image and stored it
            img_data = cv2.imread('../data/train/' + filename + '/' + label)
            X_train.append(img_data)
            y_train.append(label)

    print("done with training data")

    # load all images in a directory
    X_test = list()
    y_test = list()
    for filename in listdir('../data/test'):
        for label in listdir('../data/test/' + filename):
            # load image and stored it
            img_data = cv2.imread('../data/test/' + filename + '/' + label)
            X_test.append(img_data)
            y_test.append(label)
    
    print("done with test data")
    
    # get all 
    # X_train = training photos
    # y_train = labels for training photos
    # X_test = test phtoos
    # y_test = labels for test photos

    for size in {8, 16, 32}: # for each size of images
        resized_test = imresize(np.asarray(X_train), size) 
        print("Successflly resived test")
        resized_test_labels = imresize(y_train, size)
        print("done with resizing images!")
        for neigh in {1, 3, 6}: # for each number of neighbors
            predicted_labels = KNN_classifier(X_train, y_train, X_test, neigh) ## to do : fix this to resized!!!
            accuracy = reportAccuracy(y_test, predicted_labels, none)
            print(accuracy)
    

