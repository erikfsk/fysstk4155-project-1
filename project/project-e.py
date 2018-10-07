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
import time
import sys


class machine_learning():
	def __init__(self,func,noise_level = 0.1,degree=2,real=False):
		self.func = func
		self.degree = degree
		self.noise_level = noise_level


		#setting up data points
		self.making_x_y_z_real() if real == True else self.making_x_y_z(noise_level)

		#setting up solution variables
		self.scikit_z = None
		self.scikit_clf = None
		self.manually_z = None
		self.manually_XY = None
		self.manually_beta = None
		self.scikit_lasso_z = None
		self.scikit_lasso_clf = None
		self.manually_ridge_z = None
		self.manually_ridge_XY = None
		self.manually_ridge_beta = None
		

	def making_x_y_z_real(self):
		#setting up data points for real data
		z = imread('SRTM_data_Norway_2.tif')
		x = np.linspace(0,1,len(z[1])).reshape(len(z[1]),1)
		y = np.linspace(0,1,len(z)).reshape(len(z),1)
		self.data_points_x,self.data_points_y = np.meshgrid(x,y)
		self.data_points_z = z/np.max(z)

	def making_x_y_z(self,noise_level=None):
		#setting up data points for func
		N = 100
		noise_level = self.noise_level if noise_level is None else noise_level
		self.noise = noise_level * np.random.randn(N,1)
		x = np.sort(np.random.rand(N,1), axis=0)
		y = np.sort(np.random.rand(N,1), axis=0)
		self.data_points_x,self.data_points_y = np.meshgrid(x,y)
		z = self.func(self.data_points_x,self.data_points_y)+self.noise
		self.data_points_z = z/np.max(z)
		
	def get_x_y_z(self,x,y,z):
		#IF YOU WANT TO SAVE FLOPS!
		return self.data_points_x[::10,::10] if x is None else x,\
				self.data_points_y[::10,::10] if y is None else y,\
				self.data_points_z[::10,::10] if z is None else z
		#IF YOU DONT WANT TO SAVE FLOPs!
		# return self.data_points_x if x is None else x,\
		# 		self.data_points_y if y is None else y,\
		# 		self.data_points_z if z is None else z

	def scikit(self,degree = None,x=None,y=None,z=None):
		"""
		An OLS solution with the scikit package
		"""
		data_points_x, data_points_y, data_points_z = self.get_x_y_z(x,y,z)
		degree = self.degree if degree is None else degree
		#Scikit learn solution
		poly = PolynomialFeatures(degree=degree)
		XY = poly.fit_transform(np.array([data_points_x.ravel(),data_points_y.ravel()]).T)
		clf = linear_model.LinearRegression(fit_intercept=False)
		clf.fit(XY,data_points_z.reshape(-1,1))
		self.scikit_clf = clf
		needs_reshape = clf.predict(XY)
		self.scikit_z = needs_reshape.reshape(data_points_z.shape[0],data_points_z.shape[1])

	def scikit_lasso(self,lambda_value,degree = None,x=None,y=None,z=None):
		"""
		An Lasso solution with the scikit package
		"""
		data_points_x, data_points_y, data_points_z = self.get_x_y_z(x,y,z)
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

	def manually(self,degree = None,x=None,y=None,z=None):
		"""
		An OLS solution with self implemented method.
		"""
		data_points_x, data_points_y, data_points_z = self.get_x_y_z(x,y,z)
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

	def manually_ridge(self,lambda_value,degree = None,x=None,y=None,z=None):
		"""
		An Ridge solution with self implemented method.
		"""
		data_points_x, data_points_y, data_points_z = self.get_x_y_z(x,y,z)
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
	
	def plot(self,x=None,y=None,z=None,\
		manually_ridge_z = None, scikit_lasso_z = None, manually_z = None, scikit_z = None):
		"""
		If necessary, ability to look at the results and the data used
		All the different graphs have one square where they are drawn.
		"""
		x,y,z_noise = self.get_x_y_z(x,y,z)
		manually_ridge_z = self.manually_ridge_z if manually_ridge_z is None else manually_ridge_z
		scikit_lasso_z = self.scikit_lasso_z if scikit_lasso_z is None else scikit_lasso_z
		manually_z = self.manually_z if manually_z is None else manually_z
		scikit_z = self.scikit_z if scikit_z is None else scikit_z
		func = self.func

		# z = self.func(x, y)

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		# Plot the surface.
		# surf = ax.scatter(x, y, data_points_z, cmap=cm.coolwarm,
		# 						linewidth=0, antialiased=False)
		surf = ax.plot_surface(x, y, z_noise, cmap=cm.Greens,
								linewidth=0, antialiased=False)
		if scikit_z is not None and manually_z is not None:
			surf = ax.plot_surface(x, y+1, abs(scikit_z-manually_z), cmap=cm.Greys,
									linewidth=0, antialiased=False)
		if manually_z is not None:
			surf = ax.plot_surface(x+1, y, manually_z, cmap=cm.Blues,
								linewidth=0, antialiased=False)
			surf = ax.plot_surface(x+1, y+1, abs(manually_z-z_noise), cmap=cm.Blues,
								linewidth=0, antialiased=False)
		if manually_ridge_z is not None:
			surf = ax.plot_surface(x+2, y, manually_ridge_z, cmap=cm.Purples,
								linewidth=0, antialiased=False)
			surf = ax.plot_surface(x+2, y+1, abs(manually_ridge_z-z_noise), cmap=cm.Purples,
								linewidth=0, antialiased=False)
		if scikit_lasso_z is not None:
			surf = ax.plot_surface(x+3, y, scikit_lasso_z, cmap=cm.Reds,
								linewidth=0, antialiased=False)
			surf = ax.plot_surface(x+3,y+1, abs(scikit_lasso_z-z_noise), cmap=cm.Reds,
								linewidth=0, antialiased=False)
		# Customize the z axis.
		# ax.set_zlim(-0.10, 1.40)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

		# Add a color bar which maps values to colors.
		fig.colorbar(surf, shrink=0.5, aspect=5)

		plt.show()



	def MSE_error(self,y_computed,y_exact):
		"""
		MSE, simple calculates the MSE for the inputs, then returns MSE
		"""
		MSE = 0
		y_exact = y_exact.ravel()
		y_computed = y_computed.ravel()
		for y_computed_i,y_exact_i in zip(y_computed,y_exact):
			MSE += (y_computed_i-y_exact_i)**2
		return MSE/len(y_exact)

	def R2_error(self,y_computed,y_exact):
		"""
		R2, simple calculates the R2 for the inputs, then returns R2
		"""
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

	def get_errors(self,z=None,manually_ridge_z = None, scikit_lasso_z = None, manually_z = None, scikit_z = None):
		"""
		This functions is mostly used for printing MSE and R2 after using a method or many.
		"""
		z = self.get_x_y_z(None,None,z)[2]
		manually_ridge_z = self.manually_ridge_z if manually_ridge_z is None else manually_ridge_z
		scikit_lasso_z = self.scikit_lasso_z if scikit_lasso_z is None else scikit_lasso_z
		manually_z = self.manually_z if manually_z is None else manually_z
		scikit_z = self.scikit_z if scikit_z is None else scikit_z
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
	
	def result_var(self):
		#set need variables
		x = self.data_points_x
		y = self.data_points_y
		z = self.data_points_z
		XY = self.manually_XY
		manually_z = self.manually_z
		scikit_z = self.scikit_z
		func = self.func
		z = func(x, y)
		print()

		#READY FOR LATEX!
		for i in range(self.degree + 1):
			for j in range(self.degree + 1-i):
				print("%d%-7d&&" %(i,j),end="")

		print("\\\\")
		for i in range(len(self.manually_beta)):
			print("%-7.3f &&" % self.manually_beta[i],end="")


		sigma_2 = self.MSE_error(manually_z,z)
		beta_var = np.linalg.inv(XY.T @ XY) * sigma_2
		
		print("\\\\")
		for i in range(len(beta_var)):
			print("%-7.3f &&" % beta_var[i][i],end="")
		print("\\\\")

	def testing_degree_and_lambda(self,degrees = range(5,6),lambda_values = [0.00001],alpha_values=[0.00001],real=False):
		for degree in degrees:
			for lambda_value,alpha_value in zip(lambda_values,alpha_values):
				#simulations for different methods
				print(degree,lambda_value)
				self.scikit(degree)
				self.scikit_lasso(alpha_value,degree)
				self.manually(degree)
				self.manually_ridge(lambda_value,degree)

				#MSE and R^2 for alle methods used
				self.get_errors()

	def benchmark_testing_degree_and_lambda(self,degrees = range(5,6),lambda_values = [0.00001],alpha_values=[0.00001],real=False):
		outfile_dict = {"scikit": {2: [],3: [],4:[],5:[]},"manually": {2: [],3: [],4:[],5:[]},"ridge": {2: [],3: [],4:[],5:[]},"lasso": {2: [],3: [],4:[],5:[]}}
		n = 1000
		for i in range(n):
			for degree in degrees:
				for lambda_value,alpha_value in zip(lambda_values,alpha_values):
					#simulations for different methods
					t0 = time.time()
					self.scikit(degree)
					t1 = time.time()
					outfile_dict["scikit"][degree].append(t1-t0)

					t0 = time.time()
					self.scikit_lasso(alpha_value,degree)
					t1 = time.time()
					outfile_dict["manually"][degree].append(t1-t0)

					t0 = time.time()
					self.manually(degree)
					t1 = time.time()
					outfile_dict["lasso"][degree].append(t1-t0)

					t0 = time.time()
					self.manually_ridge(lambda_value,degree)
					t1 = time.time()
					outfile_dict["ridge"][degree].append(t1-t0)

					#MSE and R^2 for alle methods used
					# self.get_errors()
		for key_i in outfile_dict.keys():
			print(key_i)
			mean_value = n
			for key_j in outfile_dict[key_i]:
				mean_value_ny = sum(outfile_dict[key_i][key_j])/n
				if mean_value > mean_value_ny:
					mean_value = mean_value_ny
				print("%d %.2f"% (key_j,mean_value_ny/mean_value))

	def benchmark_testing_noise_level(self):
		outfile_dict = {"scikit": {0: [],0.01: [],0.2:[],0.5:[]},"manually": {0: [],0.01: [],0.2:[],0.5:[]},"ridge": {0: [],0.01: [],0.2:[],0.5:[]},"lasso": {0: [],0.01: [],0.2:[],0.5:[]}}
		n = 100
		lambda_value,alpha_value = 0.00001,0.00001
		degree = 5
		for i in range(n):
			for noise_level in [0,0.01,0.2,0.5]:
				self.making_x_y_z(noise_level=noise_level)
				#simulations for different methods
				self.scikit(degree)
				self.scikit_lasso(alpha_value,degree)
				self.manually(degree)
				self.manually_ridge(lambda_value,degree)

				z = self.get_x_y_z(None,None,None)[2]
				manually_ridge_z = self.manually_ridge_z
				scikit_lasso_z = self.scikit_lasso_z
				manually_z = self.manually_z
				scikit_z = self.scikit_z

				# outfile_dict["scikit"][noise_level].append(self.MSE_error(scikit_z,z))
				# outfile_dict["manually"][noise_level].append(self.MSE_error(manually_z,z))
				# outfile_dict["lasso"][noise_level].append(self.MSE_error(manually_ridge_z,z))
				# outfile_dict["ridge"][noise_level].append(self.MSE_error(scikit_lasso_z,z))


				outfile_dict["scikit"][noise_level].append(self.R2_error(scikit_z,z))
				outfile_dict["manually"][noise_level].append(self.R2_error(manually_z,z))
				outfile_dict["lasso"][noise_level].append(self.R2_error(manually_ridge_z,z))
				outfile_dict["ridge"][noise_level].append(self.R2_error(scikit_lasso_z,z))
				
		print(outfile_dict)
		for key_i in outfile_dict.keys():
			print(key_i)
			mean_value = n
			for key_j in outfile_dict[key_i]:
				mean_value_ny = sum(outfile_dict[key_i][key_j])/n
				if mean_value > mean_value_ny:
					mean_value = mean_value_ny
					print("%.5f" % mean_value)
				print("%d %.2f"% (key_j,mean_value_ny/mean_value))

	def results_degree(self):
		self.making_x_y_z(noise_level=0.1)
		outfile_dict = {"scikit": {},"manually": {},"ridge": {},"lasso": {}}
		values = [0.0000001,0.00001,0.001,0.1,1,2,5,10]


		for key in outfile_dict.keys():
			for i in values:
				outfile_dict[key][i] = []


		n = 1
		lambda_value,alpha_value = 0.00001,0.00001
		for i in range(n):
			for value in values:
				self.making_x_y_z(noise_level=0.1)
				#simulations for different methods
				self.scikit(5)
				self.scikit_lasso(value,5)
				self.manually(5)
				self.manually_ridge(value,5)

				z = self.get_x_y_z(None,None,None)[2]
				manually_ridge_z = self.manually_ridge_z
				scikit_lasso_z = self.scikit_lasso_z
				manually_z = self.manually_z
				scikit_z = self.scikit_z

				outfile_dict["scikit"][value].append(self.R2_error(scikit_z,z))
				outfile_dict["manually"][value].append(self.R2_error(manually_z,z))
				outfile_dict["lasso"][value].append(self.R2_error(manually_ridge_z,z))
				outfile_dict["ridge"][value].append(self.R2_error(scikit_lasso_z,z))
				
		print(outfile_dict)
		for key_i in outfile_dict.keys():
			print(key_i)
			mean_value = n
			for key_j in outfile_dict[key_i]:
				mean_value_ny = sum(outfile_dict[key_i][key_j])/n
				# if mean_value > mean_value_ny:
				# 	mean_value = mean_value_ny
				# 	print("%.5f" % mean_value)
				print("%.7f %.5f"% (key_j,mean_value_ny))

	def k_folding(self,k_parts,degree = None,lambda_value=0.001,real=False):
		degree = self.degree if degree is None else degree
		beta = 0;beta_ridge = 0;coef = 0; coef_lasso = 0;
		beta_2 = 0;beta_ridge_2 = 0;coef_2 = 0; coef_lasso_2 = 0;
		if real:
			print("Starting k-folding for Real")
			self.making_x_y_z_real()
		elif not real:
			print("Starting k-folding for test data")
			self.making_x_y_z()
		
		for i in range(k_parts-1):
			#data points for this k-fold
			x = self.data_points_x[i::k_parts] 
			y = self.data_points_y[i::k_parts] 
			z = self.data_points_z[i::k_parts]
			#function calls
			self.manually(x=x,y=y,z=z)
			self.manually_ridge(lambda_value=lambda_value,x=x,y=y,z=z)
			self.scikit(x=x,y=y,z=z)
			self.scikit_lasso(lambda_value=lambda_value,x=x,y=y,z=z)
			
			# saving predicts
			coef = coef + self.scikit_clf.coef_
			coef_lasso = coef_lasso + self.scikit_lasso_clf.coef_
			beta = beta + self.manually_beta
			beta_ridge = beta_ridge + self.manually_ridge_beta
			# saving for VAR
			coef_2 = coef_2 + self.scikit_clf.coef_**2
			coef_lasso_2 = coef_lasso_2 + self.scikit_lasso_clf.coef_**2
			beta_2 = beta_2 + self.manually_beta**2
			beta_ridge_2 = beta_ridge_2 + self.manually_ridge_beta**2

		#scaling down the beta value from k_parts-1 betas to 
		beta = beta/(k_parts-1)
		beta_ridge = beta_ridge/(k_parts-1)
		coef = coef/(k_parts-1)
		coef_lasso = coef_lasso/(k_parts-1)

		#scaling down the squared values
		beta_2 = beta_2/(k_parts-1)
		beta_ridge_2 = beta_ridge_2/(k_parts-1)
		coef_2 = coef_2/(k_parts-1)
		coef_lasso_2 = coef_lasso_2/(k_parts-1)

		print(beta)
		print(beta_2)
		print(beta_ridge)
		print(beta_ridge_2)
		print(coef)
		print(coef_2)
		print(coef_lasso)
		print(coef_lasso_2)
		#setting up data for testing
		x = self.data_points_x[(k_parts-1)::k_parts].reshape(-1,1)
		y = self.data_points_y[(k_parts-1)::k_parts].reshape(-1,1)
		z = self.data_points_z[k_parts-1::k_parts] 
		xb = self.making_xb(x,y,degree) #yields 1/100 part not 1/10

		#predict for manually solutions
		zpredict = xb.dot(beta)
		zpredict_ridge = xb.dot(beta_ridge)

		#Scikit learn solution
		poly = PolynomialFeatures(degree=degree)
		XY = poly.fit_transform(np.array([x.ravel(),y.ravel()]).T)
		clf = linear_model.LinearRegression(fit_intercept=False)
		clf.fit(XY,z.reshape(-1,1))
		#ordinary least square predict
		clf.coef_ = coef
		needs_reshape = clf.predict(XY)
		predict_scikit_z = needs_reshape.reshape(z.shape[0],z.shape[1])
		
		#lasso predict
		old_ = clf.coef_
		clf.coef_ = coef_lasso
		needs_reshape = clf.predict(XY)
		predict_scikit_lasso_z = needs_reshape.reshape(z.shape[0],z.shape[1])


		#reshape to original form
		x = x.reshape(z.shape[0],z.shape[1])
		y = y.reshape(z.shape[0],z.shape[1])
		zpredict_ridge = zpredict_ridge.reshape(z.shape[0],z.shape[1])
		predict_scikit_lasso_z = predict_scikit_lasso_z.reshape(z.shape[0],z.shape[1])
		zpredict = zpredict.reshape(z.shape[0],z.shape[1])
		predict_scikit_z = predict_scikit_z.reshape(z.shape[0],z.shape[1])

		#looking at the errors for the different methods
		if real:
			self.get_errors(z=z,manually_z = zpredict,manually_ridge_z=zpredict_ridge,\
						scikit_z=predict_scikit_z,scikit_lasso_z =predict_scikit_lasso_z)
		elif not real:
			self.get_errors(z=self.func(x,y),manually_z = zpredict,manually_ridge_z=zpredict_ridge,\
						scikit_z=predict_scikit_z,scikit_lasso_z =predict_scikit_lasso_z)
		#plot to see the difference
		# self.plot(x=x,y=y,z=z,\
		# manually_ridge_z = zpredict_ridge, scikit_lasso_z = predict_scikit_lasso_z,\
		#  manually_z = zpredict, scikit_z = predict_scikit_z)





def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4




if __name__ == '__main__':
	test = machine_learning(FrankeFunction,degree = 5,real=True)
	# test.scikit()
	# test.manually()
	# test.manually_ridge(0.00001)
	# test.result_var()
	# test.results_degree()

	for i in range(100,101):
		print("k-fold, where k is ",i)
		t0 = time.time()
		test.k_folding(i,real=True)
		t1 = time.time()

		print("time used ",t1-t0)
		print()

	# test.making_x_y_z(noise_level=0.1)
	# test.scikit()
	# test.manually()
	# test.manually_ridge(0.00001)
	# test.scikit_lasso(0.00001)
	# test.plot()
	# test.testing_degree_and_lambda()
	# test.benchmark_testing_noise_level()
	
	# test.get_errors()
	# test.var()
	
	#HAPPY TIME, FUN TIME
	# for x in range(1,6):
	# 	print((18111/2)*x**4 - 90555*x**3 + (633885/2) * x**2 - 452773*x + 217331)
	

