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


class machine_learning():
	def __init__(self,func,noise_level = 0.1,degree=2):
		self.func = func
		self.degree = degree


		#setting up data points
		z = imread('SRTM_data_Norway_2.tif')
		x = np.linspace(0,1,len(z[1])).reshape(len(z[1]),1)
		y = np.linspace(0,1,len(z)).reshape(len(z),1)
		data_points_x,data_points_y = np.meshgrid(x,y)
		self.data_points_x,self.data_points_y = data_points_x[::10,::10],data_points_y[::10,::10]
		self.data_points_z = z[::10,::10]/np.max(z)

		#setting up solution variables
		self.scikit_clf = None
		self.scikit_z = None
		self.manually_z = None
		self.manually_XY = None
		self.manually_beta = None
		self.scikit_lasso_z = None
		self.scikit_lasso_clf = None
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

	def scikit_lasso(self,lambda_value,degree = None):
		data_points_x = self.data_points_x
		data_points_y = self.data_points_y
		data_points_z = self.data_points_z
		degree = self.degree if degree is None else degree
		#Scikit learn solution
		poly = PolynomialFeatures(degree=degree)
		XY = poly.fit_transform(np.array([data_points_x.ravel(),data_points_y.ravel()]).T)
		clf = linear_model.Lasso(alpha=lambda_value,fit_intercept=False)
		clf.fit(XY,data_points_z.reshape(-1,1))
		self.scikit_lasso_clf = clf #.coef_
		needs_reshape = clf.predict(XY)
		self.scikit_lasso_z = needs_reshape.reshape(data_points_z.shape[0],data_points_z.shape[1])

	def making_xb(self,x,y,degree):
		xb = []
		# print("x = ",i,"y = ",j,"tot = ",i+j)
		for i in range(degree + 1):
			for j in range(degree + 1-i):
				xb.append(x**i * y**j)
		xb = np.concatenate(xb,axis=1)
		return xb

	def manually(self,degree = None):
		data_points_x = self.data_points_x
		data_points_y = self.data_points_y
		data_points_z = self.data_points_z
		degree = self.degree if degree is None else degree
		N1 = np.shape(data_points_z)[0]
		N2 = np.shape(data_points_z)[1]

		#Manually learning solution
		x = data_points_x.reshape(-1,1)
		y = data_points_y.reshape(-1,1)
		xb = self.making_xb(x,y,degree)
		beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(data_points_z.reshape(-1,1)) #slide 11
		zpredict = xb.dot(beta)
		self.manually_z = zpredict.reshape(N1,N2)
		self.manually_XY = xb
		self.manually_beta = beta

	def manually_ridge(self,lambda_value,degree = None):
		data_points_x = self.data_points_x
		data_points_y = self.data_points_y
		data_points_z = self.data_points_z
		degree = self.degree if degree is None else degree
		N1 = np.shape(data_points_z)[0]
		N2 = np.shape(data_points_z)[1]

		#Manually learning solution
		x = data_points_x.reshape(-1,1)
		y = data_points_y.reshape(-1,1)
		xb = self.making_xb(x,y,degree)
		I_X = np.eye(np.shape(xb)[1])
		ridge_beta = np.linalg.inv(xb.T.dot(xb) + lambda_value*I_X).dot(xb.T).dot(data_points_z.reshape(-1,1))
		zpredict = xb.dot(ridge_beta)
		self.manually_ridge_z = zpredict.reshape(N1,N2)
		self.manually_ridge_XY = xb
		self.manually_ridge_beta = ridge_beta
	
	def plot(self):
		x = self.data_points_x
		y = self.data_points_y
		z = self.data_points_z
		scikit_lasso_z = self.scikit_lasso_z
		manually_ridge_z = self.manually_ridge_z
		manually_z = self.manually_z
		scikit_z = self.scikit_z
		func = self.func

		# z = func(x, y)

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		# Plot the surface.
		# surf = ax.scatter(x, y, data_points_z, cmap=cm.coolwarm,
		# 						linewidth=0, antialiased=False)
		surf = ax.plot_surface(x, y, z, cmap=cm.Greens,
								linewidth=0, antialiased=False)
		if scikit_z is not None and manually_z is not None:
			surf = ax.plot_surface(x, y+1, abs(scikit_z-manually_z), cmap=cm.Oranges,
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
		if scikit_lasso_z is not None:
			surf = ax.plot_surface(x+3, y, scikit_lasso_z, cmap=cm.Reds,
								linewidth=0, antialiased=False)
			surf = ax.plot_surface(x+3,y+1, abs(scikit_lasso_z-z), cmap=cm.Reds,
								linewidth=0, antialiased=False)
		# Customize the z axis.
		# ax.set_zlim(-0.10, 1.40)
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
		scikit_lasso_z = self.scikit_lasso_z
		manually_z = self.manually_z
		scikit_z = self.scikit_z
		func = self.func

		# z = func(x, y)

		print("%-12s %-12s %s" % ("Method","MSE","R^2"))

		if scikit_z is not None:
			print("%-12s %-12.6f %.6f" % ("Scikit",self.MSE_error(scikit_z,z),self.R2_error(scikit_z,z)))

		if manually_z is not None:
			print("%-12s %-12.6f %.6f" % ("Manually",self.MSE_error(manually_z,z),self.R2_error(manually_z,z)))

		if manually_ridge_z is not None:
			print("%-12s %-12.6f %.6f" % ("Ridge",self.MSE_error(manually_ridge_z,z),self.R2_error(manually_ridge_z,z)))

		if scikit_lasso_z is not None:
			print("%-12s %-12.6f %.6f" % ("Lasso",self.MSE_error(scikit_lasso_z,z),self.R2_error(scikit_lasso_z,z)))
	
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


	def testing_degree_and_lambda(self,degrees = range(5,6),lambda_values = [0.00001]):
		for degree in degrees:
			for lambda_value in lambda_values:
				#simulations for different methods
				self.scikit(degree)
				self.scikit_lasso(lambda_value,degree)

				self.manually(degree)
				self.manually_ridge(lambda_value,degree)

				#MSE and R^2 for alle methods used
				print(lambda_value,degree)
				self.get_errors()

	def k_folding(self,k_parts,func=None):
		func = self.manually if func is None else func
		for i in range(k_parts):
			data_points_x = self.data_points_x[i::k_parts,i::k_parts]
			data_points_y = self.data_points_y[i::k_parts,i::k_parts]
			data_points_z = self.data_points_z[i::k_parts,i::k_parts]
			print(np.shape(data_points_x))
			print(np.shape(data_points_y))
			print(np.shape(data_points_z))




def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4




if __name__ == '__main__':
	test = machine_learning(FrankeFunction,degree = 5)
	test.testing_degree_and_lambda()
	test.k_folding(10)
	# test.plot()
	# test.get_errors()
	# test.var()
	

# 1e-05 5
# Method       MSE          R^2
# Scikit       0.010720     0.823088
# Manually     0.010720     0.823088
# Ridge        0.010720     0.823088
# Lasso        0.012273     0.797470

