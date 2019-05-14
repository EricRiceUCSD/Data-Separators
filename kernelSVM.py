import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from mnist import MNIST

testLinear = False

# Loading MNIST data and converting types
mndata = MNIST('samples')

X, y = mndata.load_training()
X = np.array(X)
y = np.array(y, dtype=np.int32)

test_X, test_y = mndata.load_testing()
test_X = np.array(test_X)
test_y = np.array(test_y, dtype=np.int32)

if testLinear:
	clf = LinearSVC(
		loss="hinge",
		max_iter=2000
	)

	Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
	for C in Cs:
		clf.C = C
		clf.fit(X, y)
		train_scores += [100 * (1 - clf.score(X, y))]
		test_scores += [100 * (1 - clf.score(test_X, test_y))]
		print(
			"For C = " +
			str(C) +
			", the training error is " +
			str(100 * (1 - clf.score(X, y))) +
			"."
		)
		print(
			"For C = " +
			str(C) +
			", the test error is " +
			str(100 * (1 - clf.score(test_X, test_y))) +
			"."
		)

else:
	clf = SVC(
		kernel="poly",
		degree=2,
		C=1.0,
		gamma="auto"
	)
	clf.fit(X, y)

	print(
		"The training error for SVC with a quadratic kernel is " +
		str(100 * (1 - clf.score(X, y))) +
		"."
	)
	print(
		"The test error for SVC with a quadratic kernel is " +
		str(100 * (1 - clf.score(test_X, test_y))) +
		"."
	)
	print(
		"There are " +
		str(sum(clf.n_support_)) +
		" support vectors."
	)
