from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from random import random, seed
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import cm
import numpy as np
import random


# print(dir(clf3))
# print(clf3.coef_)
# print(np.testing.assert_allclose(test,xb))

class machine_learning():
	def __init__(self,func,noise_level = 0.1,degree=2):
		self.func = func
		self.degree = degree

		#setting up data points
		self.noise_level = noise_level
		self.noise = noise_level * np.random.randn(100,1)

		x = np.sort(np.random.rand(100,1), axis=0)
		y = np.sort(np.random.rand(100,1), axis=0)
		self.data_points_x,self.data_points_y = np.meshgrid(x,y)
		self.data_points_z = self.func(self.data_points_x,self.data_points_y)+self.noise
		
		#setting up solution variables
		self.scikit_clf = None
		self.scikit_z = None
		self.manually_z = None
		self.manually_XY = None
		self.manually_beta = None
		self.scikit_X = None

	def scikit(self):
		data_points_x = self.data_points_x
		data_points_y = self.data_points_y
		data_points_z = self.data_points_z
		#Scikit learn solution
		poly = PolynomialFeatures(degree=self.degree)
		# XY = poly.fit_transform([data_points_x.reshape(-1,1),data_points_y.reshape(-1,1)])
		XY = poly.fit_transform(np.array([data_points_x.ravel(),data_points_y.ravel()]).T)
		# X = poly.fit_transform(data_points_x.reshape(-1,1))
		# Y = poly.fit_transform(data_points_y.reshape(-1,1))
		# XY = np.concatenate((X,Y[:,1:],X[:,1:]*Y[:,1:]),axis=1) # JEG VIL HA ALLE KOMBINASJONENE! 
		# print(np.shape(XY))
		clf = linear_model.LinearRegression(fit_intercept=False)
		clf.fit(XY,data_points_z.reshape(-1,1))
		self.scikit_clf = clf
		needs_reshape = clf.predict(XY)
		self.scikit_z = needs_reshape.reshape(data_points_z.shape[0],data_points_z.shape[1])

	def manually(self,degree = None):
		data_points_x = self.data_points_x
		data_points_y = self.data_points_y
		data_points_z = self.data_points_z
		degree = self.degree if degree is None else degree

		#Manually learning solution
		xb = []
		x = data_points_x.reshape(-1,1)
		y = data_points_y.reshape(-1,1)
		for i in range(degree + 1):
			for j in range(degree + 1-i):
				# print("x = ",i,"y = ",j,"tot = ",i+j)
				xb.append(x**i * y**j)
		xb = np.asarray(xb)
		xb = np.concatenate(xb,axis=1)
		beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(data_points_z.reshape(-1,1)) #slide 11
		# print(beta) #parametrization of the square reg
		print(beta)
		print(np.shape(beta))
		zpredict = xb.dot(beta)
		self.manually_z = zpredict.reshape(100,100)
		self.manually_XY = xb
		self.manually_beta = beta

	def plot(self):
		x = self.data_points_x
		y = self.data_points_y
		manually_z = self.manually_z
		scikit_z = self.scikit_z
		func = self.func

		z = func(x, y)

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		# Plot the surface.
		# surf = ax.scatter(x, y, data_points_z, cmap=cm.coolwarm,
		# 						linewidth=0, antialiased=False)
		surf = ax.plot_surface(x, y, z, cmap=cm.Greens,
								linewidth=0, antialiased=False)
		if scikit_z is not None:
			surf = ax.plot_surface(x+1, y, scikit_z, cmap=cm.Oranges,
									linewidth=0, antialiased=False)
		if manually_z is not None:
			surf = ax.plot_surface(x, y+1, manually_z, cmap=cm.Blues,
								linewidth=0, antialiased=False)
			surf = ax.plot_surface(x+1, y+1, abs(manually_z-z), cmap=cm.Reds,
								linewidth=0, antialiased=False)
		# Customize the z axis.
		ax.set_zlim(-0.10, 1.40)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

		# Add a color bar which maps values to colors.
		fig.colorbar(surf, shrink=0.5, aspect=5)

		plt.show()



	def MSE_error(self,y_computed,y_exact):
		MSE = 0
		y_exact = y_exact.ravel()
		y_computed = y_computed.ravel()
		for y_computed_i,y_exact_i in zip(y_computed,y_exact):
			MSE += (y_computed_i-y_exact_i)**2
		return MSE/len(y_exact)

	def R2_error(self,y_computed,y_exact):
		#ravel to two long lists
		y_exact = y_exact.ravel()
		y_computed = y_computed.ravel()

		#define sums and mean-value
		numerator = 0
		denominator = 0
		y_mean = np.mean(y_exact)

		#calculate the sums
		for y_computed_i,y_exact_i in zip(y_computed,y_exact):
			numerator += (y_computed_i-y_exact_i)**2
			denominator += (y_exact_i-y_mean)**2
		return 1 - (numerator/denominator)

	def get_errors(self):
		#set need variables
		x = self.data_points_x
		y = self.data_points_y
		z = self.data_points_z
		manually_z = self.manually_z
		scikit_z = self.scikit_z
		func = self.func

		z = func(x, y)


		if scikit_z is not None:
			print("Scikit MSE is %.6f" % self.MSE_error(scikit_z,z))
			print("Scikit R^2 is %.6f\n" % self.R2_error(scikit_z,z))

		if manually_z is not None:
			print("Manually MSE is %.6f" % self.MSE_error(manually_z,z))
			print("Manually R^2 is %.6f\n" % self.R2_error(manually_z,z))
	
	def var(self):
		#set need variables
		x = self.data_points_x
		y = self.data_points_y
		z = self.data_points_z
		XY = self.manually_XY
		manually_z = self.manually_z
		scikit_z = self.scikit_z
		func = self.func
		z = func(x, y)
		
		sigma_2 = self.MSE_error(manually_z,z)
		beta_var = np.linalg.inv(XY.T @ XY) * sigma_2



def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

if __name__ == '__main__':
	
	test = machine_learning(FrankeFunction,degree = 5)
	# test.scikit_Lasso()
	test.scikit()
	test.manually()
	#test.get_errors()
	# test.plot()
	test.var()
	












