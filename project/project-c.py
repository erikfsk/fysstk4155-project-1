from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from random import random, seed
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random


# print(dir(clf3))
# print(clf3.coef_)
# print(np.testing.assert_allclose(test,xb))

class machine_learning():
	def __init__(self,func,N = 100,noise_level = 0.1,degree=2):
		self.func = func
		self.degree = degree
		self.N = N
		#setting up data points
		self.noise_level = noise_level
		self.noise = noise_level * np.random.randn(N,1)

		x = np.sort(np.random.rand(N,1), axis=0)
		y = np.sort(np.random.rand(N,1), axis=0)
		self.data_points_x,self.data_points_y = np.meshgrid(x,y)
		self.data_points_z = self.func(self.data_points_x,self.data_points_y)+self.noise
		
		#setting up solution variables
		self.scikit_clf = None
		self.scikit_z = None
		self.manually_z = None
		self.manually_XY = None
		self.manually_beta = None
		self.manually_lasso_z = None
		self.manually_lasso_XY = None
		self.manually_lasso_beta = None
		self.manually_ridge_z = None
		self.manually_ridge_XY = None
		self.manually_ridge_beta = None
		self.scikit_X = None

	def scikit(self,degree = None):
		data_points_x = self.data_points_x
		data_points_y = self.data_points_y
		data_points_z = self.data_points_z
		degree = self.degree if degree is None else degree
		#Scikit learn solution
		poly = PolynomialFeatures(degree=degree)
		XY = poly.fit_transform(np.array([data_points_x.ravel(),data_points_y.ravel()]).T)
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
		xb = np.concatenate(xb,axis=1)
		beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(data_points_z.reshape(-1,1)) #slide 11
		# print(beta) #parametrization of the square reg
		zpredict = xb.dot(beta)
		self.manually_z = zpredict.reshape(self.N,self.N)
		self.manually_XY = xb
		self.manually_beta = beta

	def manually_ridge(self,lambda_value,degree = None):
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
		I_X = np.eye(np.shape(xb)[1])
		ridge_beta = np.linalg.inv(xb.T.dot(xb) + lambda_value*I_X).dot(xb.T).dot(data_points_z.reshape(-1,1)) #slide 11
		# print(beta) #parametrization of the square reg
		zpredict = xb.dot(ridge_beta)
		self.manually_ridge_z = zpredict.reshape(self.N,self.N)
		self.manually_ridge_XY = xb
		self.manually_ridge_beta = ridge_beta

	def manually_lasso(self,lambda_value,degree = None):
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
		I_X = np.eye(np.shape(xb)[1])
		lasso_beta = np.linalg.inv(xb.T.dot(xb) + lambda_value*I_X).dot(xb.T).dot(data_points_z.reshape(-1,1)) #slide 11
		# print(beta) #parametrization of the square reg
		zpredict = xb.dot(lasso_beta)
		self.manually_lasso_z = zpredict.reshape(self.N,self.N)
		self.manually_lasso_XY = xb
		self.manually_lasso_beta = lasso_beta

	def plot(self):
		x = self.data_points_x
		y = self.data_points_y
		manually_lasso_z = self.manually_lasso_z
		manually_ridge_z = self.manually_ridge_z
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
			surf = ax.plot_surface(x, y+1, scikit_z, cmap=cm.Oranges,
									linewidth=0, antialiased=False)
		if manually_z is not None:
			surf = ax.plot_surface(x+1, y, manually_z, cmap=cm.Blues,
								linewidth=0, antialiased=False)
			surf = ax.plot_surface(x+1, y+1, abs(manually_z-z), cmap=cm.Blues,
								linewidth=0, antialiased=False)
		if manually_ridge_z is not None:
			surf = ax.plot_surface(x+2, y, manually_ridge_z, cmap=cm.Purples,
								linewidth=0, antialiased=False)
			surf = ax.plot_surface(x+2, y+1, abs(manually_ridge_z-z), cmap=cm.Purples,
								linewidth=0, antialiased=False)
		if manually_lasso_z is not None:
			surf = ax.plot_surface(x+3, y, manually_lasso_z, cmap=cm.Reds,
								linewidth=0, antialiased=False)
			surf = ax.plot_surface(x+3,y+1, abs(manually_lasso_z-z), cmap=cm.Reds,
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
		manually_ridge_z = self.manually_ridge_z
		manually_lasso_z = self.manually_lasso_z
		manually_z = self.manually_z
		scikit_z = self.scikit_z
		func = self.func

		z = func(x, y)

		print("%-12s %-12s %s" % ("Method","MSE","R^2"))

		if scikit_z is not None:
			print("%-12s %-12.6f %.6f" % ("Scikit",self.MSE_error(scikit_z,z),self.R2_error(scikit_z,z)))

		if manually_z is not None:
			print("%-12s %-12.6f %.6f" % ("Manually",self.MSE_error(manually_z,z),self.R2_error(manually_z,z)))

		if manually_ridge_z is not None:
			print("%-12s %-12.6f %.6f" % ("Ridge",self.MSE_error(manually_ridge_z,z),self.R2_error(manually_ridge_z,z)))

		if manually_lasso_z is not None:
			print("%-12s %-12.6f %.6f" % ("Lasso",self.MSE_error(manually_ridge_z,z),self.R2_error(manually_ridge_z,z)))
	
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
		# for i in range(len(beta_var)):
		# 	print("%.3f " % beta_var[i][i],end="")



def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4




if __name__ == '__main__':
	test = machine_learning(FrankeFunction,degree = 5)
	# test.scikit_Lasso()
	for degree in [4,5,6,7,8,9]:
		for lambda_value in [0.00000001,0.00001,0.001,0.1,1]:
			test.scikit(degree)
			test.manually(degree)
			print(lambda_value,degree)
			test.manually_ridge(lambda_value,degree)
			test.manually_lasso(lambda_value,degree)
			test.get_errors()
		
	test.scikit(9)
	test.manually(9)
	test.manually_ridge(0.0000000001,9)
	test.manually_lasso(0.0000000001,9)
	test.plot()
	test.var()
	



