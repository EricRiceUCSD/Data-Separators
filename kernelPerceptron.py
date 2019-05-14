import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from matplotlib import patches


class KernelPerceptron:

	def __init__(self, X, y, Phi="Quad", s=1):
		self.X = X
		self.y = y
		self.Phi = Phi
		if self.Phi == "RBF":
			self.s = s
		self.intercept = 0
		self.alpha = np.zeros(len(X))
		self.fit()

	def k(self, x, z):
		if self.Phi == "Quad":
			return (1 + np.dot(x, z))**2
		elif self.Phi == "RBF":
			return np.exp(-(np.linalg.norm(x - z)**2) / (self.s**2))

	def decisionFunc(self, x):
		summ = 0
		for j in range(len(self.X)):
			summ += self.alpha[j] * self.y[j] * self.k(self.X[j], x)
		summ += self.intercept

		return summ

	def predict(self, x):
		return np.int32(np.sign(self.decisionFunc(x)))

	def fit(self):
		n = len(self.X)
		somePointMisclassified = True
		while (somePointMisclassified):
			pmInd = np.random.permutation(n)
			numCorrect = 0
			for i in pmInd:
				# If the point is misclassified
				if (self.y[i] * self.decisionFunc(self.X[i]) <= 0):
					# Update current point's alpha and intercept
					self.alpha[i] += 1
					self.intercept += self.y[i]
				else:
					numCorrect += 1
			if (numCorrect == n):
				somePointMisclassified = False


for filename in ["data1.txt"]:
	rawData = np.loadtxt(filename)
	X = rawData[:, :-1]
	y = np.array(rawData[:, -1], dtype=np.int32)

	n = len(X)
	d = len(X[0])

	# Setting label colors
	label_colors = np.zeros((n, 3))
	for i in range(len(y)):
		if y[i] == -1:
			label_colors[i] = (1, 0, 0)		# Red
		elif y[i] == 1:
			label_colors[i] = (0, 0, 1)  	# Blue

	red_patch = patches.Patch(color=(1, 0, 0), label="y = -1")
	blue_patch = patches.Patch(color=(0, 0, 1), label="y = 1")

	plt.legend(
		handles=[red_patch, blue_patch],
		loc="center right",
		bbox_to_anchor=(1, 0.42)
	)

	plotJustData = False
	if plotJustData:
		# Plotting the points
		plt.scatter(X[:, 0], X[:, 1], c=label_colors)

		# Labels, etc.
		plt.xlabel("$x_1$")
		plt.ylabel("$x_2$")
		plt.xlim(0, 11)
		plt.ylim(0, 11)
		plt.title("Training Data")
		plt.savefig("kernelPerceptron_data.png", dpi=500)
		plt.show()
	else:
		# Training our classifier
		clf = KernelPerceptron(X, y, Phi="Quad")

		# Plotting the decision boundaries
		h = .02  # step size in the mesh
		x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx, yy = np.meshgrid(
			np.arange(x_min, x_max, h),
			np.arange(y_min, y_max, h)
		)

		Z = np.zeros(xx.shape)
		for i in range(len(xx)):
			for j in range(len(xx[0])):
				Z[i, j] = clf.predict(np.array([xx[i, j], yy[i, j]]))
		plt.contourf(
			xx,
			yy,
			Z,
			levels=[-1, 0, 1],
			alpha=0.5,
			colors=[(1, 0, 0), (0, 0, 1)]
		)

		# Plotting the points
		plt.scatter(X[:, 0], X[:, 1], c=label_colors)

		# Labels, etc.
		plt.xlabel("$x_1$")
		plt.ylabel("$x_2$")
		plt.xlim(0, 11)
		plt.ylim(0, 11)
		plt.title("Kernel Perceptron")
		plt.savefig("kernelPerceptron.png", dpi=500)
		plt.show()

		plt.show()
