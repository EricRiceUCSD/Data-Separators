import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches


class Perceptron:

	def __init__(self, X, y, k=2):
		self.X = X
		self.y = y
		self.k = k
		self.coefs = np.zeros((k, len(X[0])))
		self.intercept = np.zeros(k)
		self.fit()

	def predict(self, x):
		yHat = np.zeros(self.k)
		for j in range(self.k):
			yHat[j] = np.dot(self.coefs[j], x) + self.intercept[j]

		return np.argmax(yHat)

	def fit(self):
		n = len(self.X)
		somePointMisclassified = True
		while somePointMisclassified:
			pmInd = np.random.permutation(n)
			numCorrect = 0
			for i in range(n):
				yHat = self.predict(self.X[pmInd[i]])
				if (yHat != self.y[pmInd[i]]):		# If the point is misclassified
					# Update current label's weight vector and intercept
					self.coefs[self.y[pmInd[i]]] += self.X[pmInd[i]]
					self.intercept[self.y[pmInd[i]]] += 1

					# Update predicted label's weight vector and intercept
					self.coefs[yHat] -= self.X[pmInd[i]]
					self.intercept[yHat] -= 1
				else:
					numCorrect += 1
			if (numCorrect == n):
				somePointMisclassified = False


rawData = np.loadtxt("data0.txt")
X = rawData[:, :-1]
y = np.array(rawData[:, -1], dtype=np.int32)

n = len(X)
d = len(X[0])

# Setting label colors
label_colors = np.zeros((n, 3))
for i in range(len(y)):
	if y[i] == 0:
		label_colors[i] = (1, 0, 0)		# Red
	elif y[i] == 1:
		label_colors[i] = (0, 0.5, 0)  	# Green
	elif y[i] == 2:
		label_colors[i] = (0, 0, 1)  	# Blue
	elif y[i] == 3:
		label_colors[i] = (0, 0, 0)  	# Black

red_patch = patches.Patch(color=(1, 0, 0), label="y = 0")
green_patch = patches.Patch(color=(0, 0.5, 0), label="y = 1")
blue_patch = patches.Patch(color=(0, 0, 1), label="y = 2")
black_patch = patches.Patch(color=(0, 0, 0), label="y = 3")

plt.legend(handles=[red_patch, green_patch, blue_patch, black_patch])

plotJustData = True
# Plotting just data vs plotting decision regions
if plotJustData:
	# Plotting the points
	plt.scatter(X[:, 0], X[:, 1], c=label_colors)

	# Labels, etc.
	plt.xlabel("$x_1$")
	plt.ylabel("$x_2$")
	plt.xlim(0, 11)
	plt.ylim(0, 11)
	plt.title("Training Data")
	plt.savefig("multiclassPerceptron_data.png", dpi=500)
	plt.show()
else:
	# Training our classifier
	clf = Perceptron(X, y, k=4)

	# Plotting the decision regions and boundaries
	h = .02  # step size in the mesh
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(
		np.arange(x_min, x_max, h),
		np.arange(y_min, y_max, h)
	)

	Z = np.zeros(xx.shape, dtype=np.int32)
	for i in range(len(xx)):
		for j in range(len(xx[0])):
			Z[i, j] = clf.predict(np.array([xx[i, j], yy[i, j]]))
	plt.contourf(
		xx,
		yy,
		Z,
		levels=[-1, 0, 1, 2, 3, 4],
		alpha=0.5,
		colors=[(1, 0, 0), (0, 0.7, 0), (0, 0, 1), (0.1, 0.1, 0.1)]
	)

	# Plotting the points
	plt.scatter(X[:, 0], X[:, 1], c=label_colors)

	# Labels, etc.
	plt.xlabel("$x_1$")
	plt.ylabel("$x_2$")
	plt.title("Multiclass Perceptron")
	plt.savefig("multiclassPerceptron.png", dpi=500)
	plt.show()
