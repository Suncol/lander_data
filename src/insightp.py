# by suncong 2021
# insight lander data proceser
#--------------------------- first import librarys ---------------------------#
import numpy as np
import pandas as pd
import os
import sys
import re
import urllib
import requests
import math as m
import scipy.interpolate as spint
# import waipy
from scipy.stats import norm
from scipy import signal
import datetime
from tqdm import tqdm

## for plotter, mind to set inline plot
# import matplotlib # mind to keep that line before other import matplotlib below
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import text
# %matplotlib inline 

# for wavelet analysis
# import scaleogram as scg # change source code in scaleogram cws.py COI_DEFAULTS dict alpha to num type "0.5"->0.5, it is a bug!
# import pywt
from joblib import Parallel, delayed

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

#--------------------------- end import librarys ---------------------------#

#--------------------------- global settings  ---------------------------#

import socket
import configparser
hostname = socket.gethostname()
# print('Runing on the server: %s' % hostname)

config =configparser.ConfigParser()
current_directory = os.path.dirname(os.path.abspath(__file__))
config.read(os.path.join(current_directory,'..','config.ini'))
config_id = hostname

try:
	n_jobs = int(config.get(config_id,'njobs'))
except:
	config_id = 'DEFAULT'
	n_jobs = int(config.get(config_id,'njobs'))

	


verbose = True

mv = -9999

dict_name = {
	"PRE":"Pressure",
	"P_20Hz":"Pressure",
	"MAT":"Temperature -Y",
	"PAT":"Temperature +Y",
	"HWS":"Wind speed",
	"MHWS":"Wind speed -Y",
	"PHWS":"Wind speed +Y",
	"WD":"Wind dir",
	"MWD":"Wind dir -Y",
	"PWD":"Wind dir +Y",
	"PE5DHZ":"ESTA pressure",
	"SAC791":"West current",
	"SAC771":"East current"
} # name of the phys fields 

dict_unit = {
	"PRE":"Pa",
	"P_20Hz":"Pa",
	"MAT":"K",
	"PAT":"K",
	"HWS":"m/s",
	"MHWS":"m/s",
	"PHWS":"m/s",
	"WD":"deg, 0N/90E",
	"MWD":"deg, 0N/90E",
	"PWD":"deg, 0N/90E",
	"PE5DHZ":"arbitrary unit",
	"SAC791":"Amp",
	"SAC771":"Amp"
}

# ttab to convert mars sol to solar longitude
ttab = [ \
	(8,300), \
	(57,330), \
	(113,0), \
	(174,30), \
	(240,60), \
	(306,90), \
	(371,120), \
	(431,150), \
	(485,180), \
	(535,210), \
	(582,240), \
	(629,270), \
	(676,300), \
	(727,330), \
	(783,360), \
	(844,0)
	] # the default sol interval is 30° ls

ttab30 = [(6, 300, 'MY34'), (57, 330, 'MY34'), (114, 0, 'MY35'), (175, 30, 'MY35'), (240, 60, 'MY35'), (307, 90, 'MY35'), (371, 120, 'MY35'), (431, 150, 'MY35'), (485, 180, 'MY35'), (535, 210, 'MY35'), (582, 240, 'MY35'), (628, 270, 'MY35'), (676, 300, 'MY35'), (726, 330, 'MY35'), (783, 0, 'MY36'), (843, 30, 'MY36'), (909, 60, 'MY36')]

# ttab15 for 15° ls interval, need to be update, but for now, we just use interval = 30° version
ttab15 = [ \
	(8,300), \
	(32,315), \
	(57,330), \
	(85,345), \
	(113,0), \
	(143,15), \
	(174,30), \
	(207,45), \
	(240,60), \
	(273,75), \
	(306,90), \
	(339,105), \
	(371,120), \
	(402,135), \
	(431,150), \
	(459,165), \
	(485,180), \
	]

# global plot settings
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
		  'figure.figsize': (15, 5),
		  'axes.labelsize': 'x-large',
		  'axes.titlesize':'x-large',
		  'xtick.labelsize':'x-large',
		  'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# choose default wavelet function for the entire notebook, other choice are left behind
# scg.set_default_wavelet('cmor1.5-1.0')

# verbose = False # if you wanna debug, please tune verbose to True to show more message!

#--------------------------- end global settings  ---------------------------#

#--------------------------- define functions ---------------------------#

def message(chars):
	if verbose:
		print ('insightp message: '+chars)

def exitmessage(chars):
	message(chars)
	sys.exit()

# convert a given martian day number (sol) into corresponding solar longitude, Ls
def mars_sol2ls(soltabin, forcecontinuity=True):
	year_day = 668.6
	peri_day = 485.35
	e_elips = 0.09340
	radtodeg = 57.2957795130823
	timeperi = 1.90258341759902

	if type(soltabin).__name__ in ['int', 'float', 'float32', 'float64']:
		soltab = [soltabin]
		solout = np.zeros([1])
	else:
		soltab = soltabin
		solout = np.zeros([len(soltab)])
	i = 0
	for sol in soltab:
		zz = (sol - peri_day) / year_day
		zanom = 2. * np.pi * (zz - np.floor(zz))
		xref = np.abs(zanom)
		#  The equation zx0 - e * sin (zx0) = xref, solved by Newton
		zx0 = xref + e_elips * m.sin(xref)
		iter = 0
		while iter <= 10:
			iter = iter + 1
			zdx = -(zx0 - e_elips * m.sin(zx0) - xref) / (1. - e_elips * m.cos(zx0))
			if (np.abs(zdx) <= (1.e-7)):
				continue
			zx0 = zx0 + zdx
		zx0 = zx0 + zdx
		if (zanom < 0.): zx0 = -zx0
		# compute true anomaly zteta, now that eccentric anomaly zx0 is known
		zteta = 2. * m.atan(m.sqrt((1. + e_elips) / (1. - e_elips)) * m.tan(zx0 / 2.))
		# compute Ls
		ls = zteta - timeperi
		if (ls < 0.): ls = ls + 2. * np.pi
		if (ls > 2. * np.pi): ls = ls - 2. * np.pi
		# convert Ls in deg.
		ls = radtodeg * ls
		solout[i] = ls
		i = i + 1
	if forcecontinuity:
		for iii in range(len(soltab) - 1):
			while solout[iii + 1] - solout[iii] > 180.:  solout[iii + 1] = solout[iii + 1] - 360.
			while solout[iii] - solout[iii + 1] > 180.:  solout[iii + 1] = solout[iii + 1] + 360.
	return solout

def insight_sol2ls(soltabin,forcecontinuity=True):
	solzero = 555
	return mars_sol2ls(soltabin+solzero,forcecontinuity=forcecontinuity)

def shiftangle(wd):
	ff = wd[np.isfinite(wd)]
	w = np.where(ff>180.)
	ff[w] = ff[w] - 360.
	wd[np.isfinite(wd)] = ff
	return wd

def avail_field(data):
	return data.dtype.names[4:]

def get_not_missing(data,code):
	return np.isfinite(data[code])

def remove_missing(data,code):
	w = get_not_missing(data,code)
	return data[code][w], data["SCLK"][w] # just return the data without missing points, SCLK dim for interp

def fill_missing(data,code,kind='linear',dt=None):
	### do not use dt>dt_file
	## first, get reference data
	w = get_not_missing(data,code)
	xref = data["SCLK"][w]
	yref = data[code][w]
	## create interpolation function
	func = spint.interpolate.interp1d(xref, yref, kind=kind, fill_value="extrapolate") # maybe the extrapolate is good
	## interpolate to full coordinate
	xnew = data['SCLK']
	## [or] to a made-up coordinate with step dt (seconds)
	if dt is not None:
		xmin = np.min(xnew)
		xmax = np.max(xnew)
		xnew = np.arange(xmin,xmax,dt)
	## perform the interpolation
	interpolated = func(xnew)
	return interpolated, xnew

def fill_missing_all(data,kind='linear'):
	codes = avail_field(data)
	for cc in codes:
		data[cc],foo = fill_missing(data,cc,kind=kind,dt=None)
	return data

def getwhere(time,mint=None,maxt=None):

	#     if mint < 0 or maxt <0: # if the mint is None, this line dead!
	#         exitmessage('time interval error, got neg time range!')

	## get indices for a time interval
	idx = None
	if mint is not None and maxt is not None:
		idx = (time > mint) * (time<maxt) # * use as "and"
	elif mint is not None:
		idx = (time > mint)
	elif maxt is not None:
		idx = (time < maxt)

	if idx is not None:
		w = np.where(idx)
	else:
		w = np.where(time >= 0)

	return w

def getres(time):
	dt = np.round(np.diff(time),2)
	res = 1./dt
	damin = np.min(res)
	damax = np.max(res)
	sps = damin
	if damin != damax:
		# the frequency in the samples are mismatched
		sps = 0.
	return sps

def ltstfloat(chartab,clock=None,indices=None):
	nt = chartab.size
	if nt == 1:
		chartab = np.array([chartab])
	if indices is None:
		## there are two types of LTST entries
		## one with simply the LTST time (MWS)
		## one with sol number + LTST time (PDS)
		zelen = len(chartab[0])
		if zelen == 8:
			indices = [0,2,3,5,6,8]
		elif zelen == 14:
			indices = [6,8,9,11,12,14]
		else:
			exitmessage("unknown LTST format!")
	tt = np.zeros((nt))
	cs = np.zeros((nt))
	n = 0

	for iii in np.arange(nt):
		ltst = chartab[iii]
		hh = float(ltst[indices[0]:indices[1]])
		mm = float(ltst[indices[2]:indices[3]])
		ss = float(ltst[indices[4]:indices[5]])
		tt[iii] = hh + mm / 60. + ss / 3600.
		## LTST is only accurate to the second
		## below the second level, if SCLK is given
		## we reconstruct the value to add in cs
		## NB: not perfect, sometimes two points overlap
		## because not regular LTST <> SCLK

		if clock is not None:
			sps = getres(clock)
			## we only correct if frequency  does not change
			if sps != 0.:
				dt = 1./sps
				if tt[iii] == tt[iii]-1:
					n = n+1
					cs[iii] = n*dt
				else:
					n = 0
	tt += cs/3600.
	return tt


def lmstfloat(chartab):
	lmst = ltstfloat(chartab,indices=[6,8,9,11,12,18])
	try:
		# one point at the beginning might not be exactly midnight
		if lmst[0] > 23.9999:
			lmst[0] = 0.
		# a couple points are the next day in the end
		# correct this to add 24 to thost LMST
		# mind to know that only works for complete sols
		w = np.where(np.diff(lmst)<0)
		if len(w) > 1:
			ind = int(w[0]+1)
			lmst[ind:] = lmst[ind:] + 24.
	except:
		pass

	return lmst



def gettime(data,timetype):
	time = data[timetype]
	# convert to float
	if timetype == 'LTST':
		time = ltstfloat(time)
	elif timetype == 'LMST':
		time = lmstfloat(time)
	return time

def makedate(chartab):
	nt = chartab.size
	tt = []
	for iii in np.arange(nt):
		char = chartab[iii]
		yy = int(char[0:4]) #; print yy
		mm = int(char[5:7]) #; print mm
		dd = int(char[8:10]) #; print dd
		hh = int(char[11:13]) #; print hh
		mi = int(char[14:16]) #; print mi
		ss = int(char[17:19]) #; print ss
		tt.append(datetime.datetime(yy,mm,dd,hh,mi,ss))
	return tt

def reduced_data(data,mint=None,maxt=None,timetype='LTST'):
	# get a data object reduced to the indicated time interval

	# 1. get times (converted correctly)
	time = gettime(data,timetype)
	# 2. get indices
	w = getwhere(time,mint=mint,maxt=maxt)
	# 3. build new darray
	ref = data["SCLK"][w]
	dataout = np.ndarray(ref.shape, dtype=data.dtype)
	# 4. fill new ndarray
	codes = data.dtype.names
	for code in codes:
		dataout[code] = data[code][w]
	return dataout

def dir_name_sort(dir_name_list,des=False):
	dir_name_list.sort(key=lambda x: int(x.split('_')[-1]),reverse=des)
	return dir_name_list

def file_name_sort(file_name_list,des=False):
	file_name_list.sort(key=lambda x: int(x.split('_')[-2]),reverse=des)
	return file_name_list

def get_data_dirs(data_path):
	dirs = [data_dir for data_dir in os.listdir(data_path) if os.path.os.path.isdir(os.path.join(data_path,data_dir))]
	reg = re.compile(r'sol_\d{4}_\d{4}')
	return dir_name_sort(list(filter(reg.match,dirs)))

def csv2npy(data_path,data_dirs,npyfolder,data_type):
	if data_type == 'PS':
		reg = re.compile(r'ps_calib\w{0,5}_\d{4}_\d{2}\.csv')
	elif data_type == 'TWINS':
		reg = re.compile(r'twins_model\w{0,5}_\d{4}_\d{2}\.csv')
	if "twins" in data_path: #
		dtype = [('AOBT', '<f8'), \
				 ('SCLK', '<f8'), ('LMST', 'S18'), \
				 ('LTST', 'S14'), ('UTC', 'S22'), \
				 ('HWS', '<f8'), ('VERTICAL_WIND_SPEED', '?'), \
				 ('WD', '<f8'), ('WIND_FREQUENCY', '<f8'), \
				 ('WS_OPERATIONAL_FLAGS', '<i8'), ('MAT', '<f8'), \
				 ('BMY_AIR_TEMP_FREQUENCY', '<f8'), \
				 ('BMY_AIR_TEMP_OPERATIONAL_FLAGS', '<i8'), \
				 ('PAT', '<f8'), ('BPY_AIR_TEMP_FREQUENCY', '<f8'), \
				 ('BPY_AIR_TEMP_OPERATIONAL_FLAGS', '<i8')]
	else:
		dtype = None


	# for file_path in tqdm(data_paths,leave=False):
	def save_to_csvfile(file_path):
		npy_file = file_path[:-3]+'npy'
		if not os.path.exists(os.path.join(npyfolder,npy_file)):
			try:
				data = np.genfromtxt(os.path.join(data_path,adir,file_path), \
										dtype=dtype,names=True,delimiter=',', \
										filling_values=(mv))
# 					message('Finished '+file_path+' in '+adir)
			except:
				message(' File is not found or not readable')
				data = None
				np.save(os.path.join(npyfolder,npy_file),data)
				message("saved a VOID Numpy binary file: " + npy_file)
				return
			for name in data.dtype.names:
				try:
					w = np.where(data[name] == -9999)
					data[name][w] = np.nan
				except:
					pass

			if "PRESSURE" in data.dtype.names:
				### PRESSURE  ----> PRE
				data.dtype.names = ('AOBT', 'SCLK', 'LMST', 'LTST', 'UTC', \
									'PRE', 'PRESSURE_FREQUENCY', \
									'PRESSURE_TEMP', 'PRESSURE_TEMP_FREQUENCY')

			if "HORIZONTAL_WIND_SPEED" in data.dtype.names:
				### HORIZONTAL_WIND_SPEED --> HWS
				### WIND_DIRECTION --> WD
				### BMY_AIR_TEMP --> MAT
				### BPY_AIR_TEMP --> PAT
				data.dtype.names = ('AOBT', 'SCLK', 'LMST', 'LTST', 'UTC', \
									'HWS', 'VERTICAL_WIND_SPEED', \
									'WD', 'WIND_FREQUENCY', 'WS_OPERATIONAL_FLAGS', \
									'MAT', 'BMY_AIR_TEMP_FREQUENCY', 'BMY_AIR_TEMP_OPERATIONAL_FLAGS', \
									'PAT', 'BPY_AIR_TEMP_FREQUENCY', 'BPY_AIR_TEMP_OPERATIONAL_FLAGS')

			if "HWS" in data.dtype.names:
				if data['HWS'].dtype == 'bool':
					data['HWS'] = False

			codes = avail_field(data)

			## convert wind direction to interval [-180, 180]
			if "WD" in codes:
				data["WD"] = shiftangle(data["WD"])
			if "PWD" in codes:
				data["PWD"] = shiftangle(data["PWD"])
			if "MWD" in codes:
				data["MWD"] = shiftangle(data["MWD"])

			## save the normal data to npy file
			np.save(os.path.join(npyfolder,npy_file),data)
# 				message("saved a numpy binary file: " + npy_file)

	for i,adir in enumerate(data_dirs):
		file_paths = [file_path for file_path in os.listdir(os.path.join(data_path,adir)) \
					  if os.path.isfile(os.path.join(data_path,adir,file_path))]
		data_paths = file_name_sort(list(filter(reg.match,file_paths)))

		Parallel(n_jobs=n_jobs)(delayed(save_to_csvfile)(file_path) for file_path in tqdm(data_paths))

# 		for file_path in tqdm(data_paths,leave=False):
# 			npy_file = file_path[:-3]+'npy'
# 			if not os.path.exists(os.path.join(npyfolder,npy_file)):
# 				try:
# 					data = np.genfromtxt(os.path.join(data_path,adir,file_path), \
# 										 dtype=dtype,names=True,delimiter=',', \
# 										 filling_values=(mv))
# # 					message('Finished '+file_path+' in '+adir)
# 				except:
# 					message(' File is not found or not readable')
# 					data = None
# 					np.save(os.path.join(npyfolder,npy_file),data)
# 					message("saved a VOID Numpy binary file: " + npy_file)
# 					return
# 				for name in data.dtype.names:
# 					try:
# 						w = np.where(data[name] == -9999)
# 						data[name][w] = np.nan
# 					except:
# 						pass

# 				if "PRESSURE" in data.dtype.names:
# 					### PRESSURE  ----> PRE
# 					data.dtype.names = ('AOBT', 'SCLK', 'LMST', 'LTST', 'UTC', \
# 										'PRE', 'PRESSURE_FREQUENCY', \
# 										'PRESSURE_TEMP', 'PRESSURE_TEMP_FREQUENCY')

# 				if "HORIZONTAL_WIND_SPEED" in data.dtype.names:
# 					### HORIZONTAL_WIND_SPEED --> HWS
# 					### WIND_DIRECTION --> WD
# 					### BMY_AIR_TEMP --> MAT
# 					### BPY_AIR_TEMP --> PAT
# 					data.dtype.names = ('AOBT', 'SCLK', 'LMST', 'LTST', 'UTC', \
# 										'HWS', 'VERTICAL_WIND_SPEED', \
# 										'WD', 'WIND_FREQUENCY', 'WS_OPERATIONAL_FLAGS', \
# 										'MAT', 'BMY_AIR_TEMP_FREQUENCY', 'BMY_AIR_TEMP_OPERATIONAL_FLAGS', \
# 										'PAT', 'BPY_AIR_TEMP_FREQUENCY', 'BPY_AIR_TEMP_OPERATIONAL_FLAGS')

# 				if "HWS" in data.dtype.names:
# 					if data['HWS'].dtype == 'bool':
# 						data['HWS'] = False

# 				codes = avail_field(data)

# 				## convert wind direction to interval [-180, 180]
# 				if "WD" in codes:
# 					data["WD"] = shiftangle(data["WD"])
# 				if "PWD" in codes:
# 					data["PWD"] = shiftangle(data["PWD"])
# 				if "MWD" in codes:
# 					data["MWD"] = shiftangle(data["MWD"])

# 				## save the normal data to npy file
# 				np.save(os.path.join(npyfolder,npy_file),data)
# # 				message("saved a numpy binary file: " + npy_file)



# get data in a sol, mind to preprocess the data file to npy
def getsol(sol,npyfolder,data_bundle,event=False):
	'''
	var is the param name we wanna to perform
	'''
	suffix = ""
	if event:
		suffix = "event" # load event only files
	if data_bundle == 'ps':
		prefix = "ps_calib"+suffix
	elif data_bundle == 'twins':
		prefix = "twins_model"+suffix

	reg = re.compile(r"%s" % prefix + "_%s_\d{2}.npy" %("{0:0>4}".format(sol)))
	npyfiles = [f for f in os.listdir(npyfolder) if os.path.isfile(os.path.join(npyfolder, f))]

	sol_files = list(filter(reg.match,npyfiles)) # the file number may be larger than 1

	return np.load(os.path.join(npyfolder,sol_files[0])) # return one of the results, maybe change it later

def ratiodd(data,mean=13.,std=1.8,code='PRE'):
	# calculate a ratio to normalize vortex counts for incomplete sols
	# ratio == 1 if a complete sol

	## calculate integral of PDF for local times covered assumed gaussian
	## use LMST because it has decimals of seconds contrary to LTST
	## ... but mean and std guessed from LTST
	x = lmstfloat(data['LMST'])
	try:
		prod = data[code] * x
		x = x[np.isnan(prod) == False]
	except:
		# either PRE is not in file or no valid values for PRE are found
		return 0 # so return ratio equals to zero

	f_x = norm.pdf(x,mean,std)

	integrand = f_x * np.gradient(x)
	## hack for param
	fac = 100.
	w = np.where(integrand < (fac * np.mean(integrand)))
	integrand = integrand[w]
	#     x = x[w]
	#     f_x = f_x[w]
	## end hack for param

	ratio = np.round(np.sum(integrand),decimals=3)

	return ratio

## signal smooth using convolution kernal
def smooth1d(x,window_len=11,window='hanning'):
	if x.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")

	if x.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")


	if window_len<3:
		return x


	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


	s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')

	y=np.convolve(w/w.sum(),s,mode='valid')
	# need to know that the len of y do not equals to x! the transformation of convolve change the size
	return y


def detrendsmooth(data,window):
	# the window in this func is the number of the sample points
	datasmooth = smooth1d(data,window_len=int(window))
	nt = datasmooth.size
	ntt = data.size
	nn = (nt-ntt) // 2
	if nn == 2:
		datasmooth = datasmooth[2:nt-2]
	elif nn != 0:
		datasmooth = datasmooth[nn:nt-nn-1]
	if data.size != datasmooth.size:
		message("something wrong in the detrendsmooth func, the data size != datasmooth size!")
	detrend = data-datasmooth
	return detrend, datasmooth

def smoothresample(data,code,kind='linear',freq=20,window=100,complete=False,reinterpolate=True):
	## freq in HZ, window in seconds

	## this hack allows to get smooth+detrend versions
	## - when there is a cut in the middle of the sample
	## - when there is a high-frequency ERP inserted
	## - at a constant frequency (provided by freq, usually the highest)

	## 1. get interpolation at hi-frequency
	hif, hix = fill_missing(data,code,dt=1./freq,kind=kind)

	## 2. smooth and detrend this interpolation at high-frequency
	swin = window * freq
	hid, his = detrendsmooth(hif,swin)

	## 3. re-interpolate results to original time series, since data are interpolated in the fill_missing func
	if reinterpolate:
		funcd = spint.interpolate.interp1d(hix, hid, kind=kind, fill_value='extrapolate')
		funcs = spint.interpolate.interp1d(hix, his, kind=kind, fill_value="extrapolate")
		if not complete:
			w = get_not_missing(data,code)
			xnew =  data["SCLK"][w]
		else:
			xnew = data["SCLK"]
		hid, his = funcd(xnew), funcs(xnew)

		## the points within boundaries +/- window//2 should be removed
		sec = xnew
		nn = sec[0]+(window/2)
		xx = sec[-1]-(window/2)
		q = getwhere(sec,maxt=nn)
		hid[q] = np.nan
		his[q] = np.nan
		q = getwhere(sec,mint=xx)
		hid[q] = np.nan
		his[q] = np.nan

	return hid, his


# get multisol data 
# the output data is for 1d plot 
def multisol(solini=66,solsol=20,code="HWS",detrend=False,freq=0.1,win=None,timetype="LMST",ttinter=[[0,24]],compute='mean',facselec=20./24.,npyfolder=None,ymin=0,ymax=10):
	# plot settings
	#     ymin = 0
	#     ymax = 30
	addlspos = 1.05
	xtickstep = 20
	fname = './img/'+code+'_'+str(solini)+'_'+str(solini+solsol)+'_'+compute+'.png'
	fig, ax = plt.subplots(figsize=(25,10))
	color_num = len(ttinter) # set the number of the color number equals to the number of time intervals
	colors = cm.rainbow(np.linspace(0, 1, color_num))
	labels = ['sol lt: '+str(tt[0])+'-'+str(tt[1]) for tt in ttinter]
	scatter_xbin = [np.array([]) for _ in range(color_num)] # bins to put scatter points
	scatter_ybin = [np.array([]) for _ in range(color_num)]

	if code == "PRE":
		data_bundle = 'ps'
	elif code in ["HWS","WD","MAT","PAT"]:
		data_bundle = 'twins'

	outx = np.array([])
	outy = np.array([])

	# define some dirty things
	if len(ttinter) == 1:
		ttmin, ttmax = ttinter[0][0], ttinter[0][1]
		lentt = ((ttmax-ttmin))*facselec
	else:
		ttmin, ttmax = ttinter[0][0], ttinter[-1][1] # original is 0., 24.
		#         ttmin, ttmax = 0., 24. # mind to check this
		lentt = ((ttmax-ttmin))*facselec # may cause something wrong
		lentt = 0

	for sol in range(solini, solini+solsol+1):
		try:
			## get the data
			data = getsol(sol,npyfolder,data_bundle)
			ratio = ratiodd(data,mean=13.,std=1.8,code=code)
			ratio = 1.

			if ratio > 0.7: # hard code to 0.7
				## get time
				time = gettime(data,timetype)
				nm = get_not_missing(data,code)
				time = time[nm]

				## get the intersection interval request vs. data
				tmin = np.min(time)
				tmax = np.max(time)
				zetmin = np.max([tmin,ttmin])
				zetmax = np.min([tmax,ttmax])
				dtime = zetmax - zetmin

				if dtime > lentt:
					message("SELECTED sol=%i LT=[%.1f,%.1f]" % (sol,tmin,tmax))

					## smooth resample (process all sol long to minimize the adverse impact of time cuts)
					if win is not None:
						dpp, spp = smoothresample(data,code,freq=freq,window=win,complete=False)
					else:
						dpp, spp = data[code][nm], data[code][nm]

					## loop for every time interval in the ttinter
					for itt,tt in enumerate(ttinter):
						### select locally (in the bins) if we have enough data
						ttminloc, ttmaxloc = tt[0],tt[1]
						lenttloc = (ttmaxloc - ttminloc) *facselec
						zetminloc = np.max([tmin,ttminloc])
						zetmaxloc = np.min([tmax,ttmaxloc])
						dtimeloc = zetmaxloc - zetminloc

						if dtimeloc > lenttloc:
							w = getwhere(time,mint=tt[0],maxt=tt[1]) # get the time intercal idx

							outyloc = spp[w]
							if detrend:
								outyloc = dpp[w]
							outxloc = sol + time[w] / 24.

							if len(outyloc) > 0:
								if compute is not None:
									# calculate diff op for outyloc
									if compute == "std":
										outyloc = np.std(outyloc)
									elif compute == "mean":
										outyloc = np.mean(outyloc)
									elif compute == "max":
										outyloc = np.max(outyloc)
									elif compute == "min":
										outyloc = np.min(outyloc)
									elif compute == "var":
										outyloc = np.var(outyloc,ddof=1)
								# outxloc, just mean it
								outxloc = np.mean(outxloc)

								# put the single point result to the scatter bin
								scatter_ybin[itt] = np.append(scatter_ybin[itt],outyloc)
								scatter_xbin[itt] = np.append(scatter_xbin[itt],outxloc)

								outy = np.append(outy,outyloc)
								outx = np.append(outx,outxloc)
						else:
							message("NOT SELECTED LT=[%.1f,%.1f]" % (zetminloc,zetmaxloc))
			else:
				message("NOT SELECTED sol=%i LT=[%.1f,%.1f]" % (sol,tmin,tmax))

		except:
			pass

	# run plotting code separately can make it much quicker!
	for ibin in range(color_num):
		# plot scatter
		plt.scatter(scatter_xbin[ibin],scatter_ybin[ibin],color=colors[ibin],label=labels[ibin])

	plt.xticks(np.arange(0,solsol//xtickstep + 2)*xtickstep+np.round(solini,-1))
	plt.ylim(ymin,ymax)
	ax.tick_params(axis='both', which='major', labelsize=16)
	ax.tick_params(axis='both', which='minor', labelsize=14)
	plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',framealpha=0.6,fontsize=18)
	plt.title(compute+' of '+dict_name[code]+ ' from insight sol '+str(solini)+' to '+str(solini+solsol),y=-0.14,fontsize=25)
	plt.xlabel('sol',fontsize=20)
	plt.ylabel('%s (%s)' %(dict_name[code],dict_unit[code]),fontsize=20)


	# add solar longitude axis
	font = {'family': 'serif',
			'color':  'darkred',
			'weight': 'normal',
			'size': 16,
			}
	for tt in ttab:
		sol, ls = tt
		if solini < sol < solini+solsol-1:
			pos = ymax * addlspos
			text(sol,pos,r'$L_s=%i^{\circ}$'%(ls),color='m', \
				 horizontalalignment='center',verticalalignment='center',fontdict=font)
			plt.vlines(sol,ymin,ymax,color='m')

	plt.savefig(fname, dpi=600, format='png' ,bbox_inches='tight')
	# maybe we need a plt.close is the ploting a lot of pictures

	return outy, outx

def getinsol(sol,code="HWS",mint=None,maxt=None, \
			 timetype='LTST',freq=10.,window=None,addsmooth=False, \
			 npyfolder=None):

	# prepared for data bundle
	if code == "PRE":
		data_bundle = 'ps'
	elif code in ["HWS","WD","MAT","PAT"]:
		data_bundle = 'twins'

	# get the data from
	data = getsol(sol,npyfolder,data_bundle)

	## 1. get times
	time = gettime(data,timetype)

	### since the rmdisorder can lead to unsuitable slice, we add these code here
	# TBD: wrap to a function
	# a couple points are the next day in the end
	# correct this to add 24 to thost LMST
	# mind to know that only works for complete sols
	w = np.where(np.diff(time) < 0)
	if len(w) > 0:
		w = np.append(w,len(time)-1)
		w = np.insert(w,0,0)
		time_range = np.diff(w).tolist()
		idx = time_range.index(max(time_range))
		# time[w[idx]+1:w[idx+1]] = time[w[idx]+1:w[idx+1]] +24.
		time[:w[idx]+1] += 24
		time[w[idx+1]:] += 24

	### since the rmdisorder can lead to unsuitable slice, we add these code here

	## 2. get indices
	w =getwhere(time,mint=mint,maxt=maxt)

	## 3. reduce time
	time = time[w]

	## 4. min/max
	xmin, xmax = np.min(time), np.max(time)
	xmin, xmax = np.max([mint,xmin]), np.min([maxt,xmax])

	### select time interval
	# field = data[code][w] # put it in the if branch
	# field = data[w]
	if freq is None:
		freq = np.int64(np.round(1./getmode(np.diff(data['SCLK'])))) # get the real freq from the time series
	### if window is provided, smooth and detrend
	if window is not None:
		dpp, spp = smoothresample(data,code,freq=freq,window=window,complete=True)
		sfield = spp[w]
		if addsmooth: # if we just wanna smoothed data
			field = spp[w]
		else:
			field = dpp[w]
	else:
		field = data[code][w]

	### get finite values
	ww = np.isfinite(field)

	y = field[ww]
	x = time[ww]
	xref = data['SCLK'][w][ww]

	return y, x, xref

# plot the data field in a sol, with wavelet analysiss
# mind to assign period boundary if set wavelet == True
def plotinsol(y,x,xref,code,sol,datatype='detrended',timetype='LTST',plottype='scatter',ymin=None,ymax=None,color='m', \
			  wavelet=False,wavelet_plot=True,lowperiod=None,highperiod=None,return_wavelet=False):
	fname = './img/'+datatype+'-'+code+'_'+str(sol)+'_'+str(np.round(np.min(x),0))+ \
			str(np.round(np.max(x),0))+'.png'

	xtickstep = 2 #0.25

	# data preprocessing
	# xref, x, y = rmdisorder(xref, x, y) # remove time disordered samples from the input

	# # begin plot
	# if plottype == 'scatter':
	# 	plt.scatter(x,y,color=color)
	# elif plottype == 'line':
	# 	plt.plot(x,y,color=color)
	#
	# ## incase that the xtickstep is relatively small insol
	# ## so here we multiply 100 on both de/numerator
	# fig, (ax1,ax2) = plt.subplots(2,figsize=(25,10))
	# xticks_nums = np.arange(0,((np.max(x)-np.min(x))*100)//(xtickstep*100) + 2)*xtickstep+np.round(np.min(x),0)
	# ax1.set_xticks(xticks_nums)
	# ax1.set_ylim(ymin,ymax)
	# ax1.tick_params(axis='both', which='major', labelsize=16)
	# ax1.tick_params(axis='both', which='minor', labelsize=14)
	# ax1.set_title(datatype+'-'+dict_name[code]+ ' in insight sol: '+str(sol)+' intra-day change ',y=-0.14,fontsize=25)
	# ax1.set_xlabel('Mars '+timetype+'(hours)',fontsize=20)
	# ax1.set_ylabel('%s (%s)' %(dict_name[code],dict_unit[code]),fontsize=20)
	# # # fig.tight_layout()

	if wavelet:
		# #         message("wavelet analysis on!")
		# # ------old version of wavelet analysis------#
		# scales = scg.periods2scales(np.arange(lowperiod,highperiod))
		# ax2 = scg.cws(xref,y,scales=scales,cmap='jet',xlabel='Time',ylabel='Period (Seconds)', \
		# 			  title='wavelet analysis',yscale='log',figsize=(25,10))
		#
		# # change the x axis from SCLK time to timetype with time var-x
		# func = spint.interpolate.interp1d(x, xref, kind='nearest', fill_value="extrapolate")
		# xrefticks_nums = func(xticks_nums)
		# ax2.set_xticks(xrefticks_nums)
		# ax2.set_xticklabels([str(xtick) for xtick in xticks_nums])
		#
		#
		# # save the plot to file with high dpi
		# plt.savefig(fname, dpi=600, format='png' ,bbox_inches='tight')
		# # ------old version of wavelet analysis --------#

		# #----------- wavelet analysis using waipy --------------#
		# import waipy
		# ynorm = waipy.normalize(y)
		# alpha = np.corrcoef(ynorm[0:-1],ynorm[1:])[0,1]
		# message("Lag-1 autocorrelation = {:4.2f}".format(alpha))
		#
		# xdelta = xref - xref[0]
		# pad = 1
		# dj = 0.25
		# dt = np.mean(np.diff(xref)) # just mean the xref different
		# s0 = 240*dt
		# j1 = 5./dj
		# lag1 = 0.72
		# param = 6
		# mother = 'Morlet'
		# print('before cwt')
		# result = waipy.cwt(ynorm, dt, pad, dj, s0, j1, lag1, param, mother='Morlet', name=str(sol))
		# print('after cwt')
		# var = 'wavelet - '+dict_name[code]+ ' in insight sol: '+str(sol)+' intra-day change '
		# dtmin = dt / 8
		# waipy.wavelet_plot(var, x, ynorm, dtmin, result)


		# # save the plot to file with high dpi
		# plt.savefig(fname, dpi=600, format='png' ,bbox_inches='tight')
		# # ----------- wavelet analysis using waipy --------------#

		# # ----------- lastest version of wavelet analysis--------- #
		import matplotlib.pylab as plt
		import matplotlib.ticker as ticker
		from matplotlib.gridspec import GridSpec
		from waveletFunctions import wave_signif, wavelet

		ynorm = normalize(y)
		std = np.std(ynorm,ddof=1)

		n = len(ynorm)
		# dt = getmode(np.diff(xref)) # just use np.min can also be fun
		dt = getmode(np.diff(xref)) # just mean the xref different
		time = xref - xref[0] # change to time delta

		# wavelet settings
		pad = 1 # pad the time series with zeros (recommended)
		dj = 0.25
		if dt <= 0.2: # set different scale with different dt
			s0 = 900*dt
			dj = 0.05
		elif dt >= 0.4:
			s0 = 180*dt
		j1 = 8/dj
		doff = [2000,9000]
# 		doff = [100,5000]
		lag1 = 0.72 # lag-1 autocorrelation for red noise background
		mother = 'MORLET'
		Cdelta = 0.776  # this is for the MORLET wavelet

		# wavelet transform
		wave, period, scale, coi = wavelet(ynorm, dt, pad, dj, s0, j1, mother)
# 		print(period)
		power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
		# import pywt
		# scale = np.logspace(3.0,5.0,num=40)
		# [cfs, frequencies] = pywt.cwt(ynorm, scale, wavelet, dt)
		# power = (np.abs(cfs))**2
		# period = 1./frequencies
		global_ws = (np.sum(power, axis=1) / n)  # time-average over all times

		## significance levels
		signif = wave_signif(np.std(y), dt=dt, sigtest=0, scale=scale,lag1=lag1, mother=mother)
		### expand signif --> (J+1)x(N) array
		sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
		sig95 = power / sig95  # where ratio > 1, power is significant

		## Global wavelet spectrum & significance levels:
		dof = n - scale  # the -scale corrects for padding at edges
		global_signif = wave_signif(np.std(y), dt=dt, scale=scale, sigtest=1,lag1=lag1, dof=dof, mother=mother)

		# Scale-average between period
		avg = np.logical_and(scale >= doff[0], scale < doff[1])
		# expand scale --> (J+1)x(N) array
		scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
		scale_avg = power / scale_avg  # [Eqn(24)]
		scale_avg = dj * dt / Cdelta * sum(scale_avg[avg, :])  # [Eqn(24)]

		scaleavg_signif = wave_signif(std, dt=dt, scale=scale, sigtest=2,lag1=lag1, dof=doff, mother=mother)

		
		if wavelet_plot:
			# --- Plot time series
			fig = plt.figure(figsize=(9, 10))
			gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
			plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
								wspace=0, hspace=0)
			plt.subplot(gs[0, 0:3])
			plt.plot(x, ynorm, 'k')
			# plt.xlim(xlim[:])
			xticks_nums = np.arange(0, ((np.max(x) - np.min(x)) * 100) // (xtickstep * 100) + 2) * xtickstep + np.round(
				np.min(x), 0)
			xticks_nums = np.arange(0,26,2)
			plt.xticks(xticks_nums)
			plt.xlim(0,24)
			plt.ylim(ymin,ymax)
	# 		plt.xlabel('Time')
			plt.xlabel('LTST')
			plt.ylabel('%s (%s)' %(dict_name[code],dict_unit[code]))
			plt.title('a) '+dict_name[code]+ ' in insight sol: '+str(sol)+' intra-day change ')

			# plt.text(time[-1] + 35, 0.5, 'Wavelet Analysis\nC. Torrence & G.P. Compo\n'
			# 							 'http://paos.colorado.edu/\nresearch/wavelets/',
			# 		 horizontalalignment='center', verticalalignment='center')

			# --- Contour plot wavelet power spectrum
			# plt3 = plt.subplot(3, 1, 2)
			plt3 = plt.subplot(gs[1, 0:3])
			# levels = [0, 0.5, 1, 2, 4, 999]
			# levels = []
			# *** or use 'contour'
			# CS = plt.contourf(time, period, power, len(levels))
	# 		CS = plt.contourf(time, period, np.log10(power),levels=np.arange(-5,5.5,0.5))
			CS = plt.contourf(x, period, np.log10(power),levels=np.arange(-5.,5.1,0.1))

			# im = plt.contourf(CS, levels=levels,
			# 				  colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
	# 		plt.xlabel('Time (s)')
			plt.xlabel('LTST')
			plt.ylabel('Period (s)')
			plt.xticks(xticks_nums)
			plt.title('b) Wavelet Power Spectrum')
			# plt.xlim(xlim[:])
			# 95# significance contour, levels at -99 (fake) and 1 (95# signif)
	# 		plt.contour(time, period, sig95, [-99, 1], colors='k')
			plt.contour(x, period, sig95, [-99, 1], colors='k')
			# cone-of-influence, anything "below" is dubious
	# 		plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none",
	# 						 edgecolor="#00000040", hatch='x')
			plt.fill_between(x, coi * 0 + period[-1], coi, facecolor="none",
				 edgecolor="#00000040", hatch='x')
	# 		plt.plot(time, coi, 'k')
			plt.plot(x,coi,'k')
			# format y-scale
			plt3.set_yscale('log', base=10, subs=None)
	# 		plt.ylim([np.min(period), np.max(period)])
			plt.xlim(0,24)
# 			plt.ylim(100,5000)
			plt.ylim(doff)
			ax = plt.gca().yaxis
			ax.set_major_formatter(ticker.ScalarFormatter())
			plt3.ticklabel_format(axis='y', style='plain')
			plt3.invert_yaxis()
			# set up the size and location of the colorbar
			# position=fig.add_axes([0.5,0.36,0.2,0.01])
			# plt.colorbar(im, cax=position, orientation='horizontal')
			#   , fraction=0.05, pad=0.5)

			# plt.subplots_adjust(right=0.7, top=0.9)

			# --- Plot global wavelet spectrum
			plt4 = plt.subplot(gs[1, -1])
			# plt.plot(global_ws, period)
			# plt.plot(global_signif, period, '--')
			plt.semilogx(global_ws,period)
			plt.semilogx(global_signif,period,'--')
			plt.xlabel('Power')
			plt.title('c) Global Wavelet Spectrum')
			plt.xlim([np.min(global_ws), 1.25 * np.max(global_ws)])
			# format y-scale
			plt4.set_yscale('log', base=10, subs=None)
	# 		plt.ylim([np.min(period), np.max(period)])
			plt.ylim(0,24)
# 			plt.ylim(100,5000)
			plt.ylim(doff)
			ax = plt.gca().yaxis
			ax.set_major_formatter(ticker.ScalarFormatter())
			plt4.ticklabel_format(axis='y', style='plain')
			plt4.invert_yaxis()

			# --- Plot 2--8 yr scale-average time series
			plt.subplot(gs[2, 0:3])
	# 		plt.plot(time, scale_avg, 'k')
			plt.plot(x, scale_avg, 'k')
			# plt.xlim(xlim[:])
	# 		plt.xlabel('Time (s)')
			plt.xlim(0,24)
			plt.xlabel('LTST')
			plt.ylabel('Avg variance')
			plt.xticks(xticks_nums)
	# 		plt.title('d) 100-7000 s Scale-average Time Series')
			plt.title('d) Scale-average Time Series')
			# plt.plot(xlim, scaleavg_signif + [0, 0], '--')

			# plt.show()

			#save the plot to file with high dpi
			plt.savefig(fname, dpi=600, format='png' ,bbox_inches='tight')
			plt.close()
			# # ----------- lastest version of wavelet analysis--------- #

		if return_wavelet:
# 			powersum = np.nansum(power[:]) # sum of the power in the chosen frequency range as a weight between sols
# 			return time, period, np.log10(power*powersum/(np.nanmax(x)-np.nanmin(x))*dt)-6 #np.log10(power) # mind the returned power is logged scale
			powersum = np.nansum((global_ws*np.gradient(period)*dt*dt)[:])
			return time, period, np.log10(power*powersum)-6.5#5.5


# test if the time series are stationary using augmented Dickey-Fuller test
## the dickey-fuller test tests the null hypothesis that a unit root is present in an 
## autoregressive time series model
# def is_stationary(y,freq=None,window=None,detrend=False): # just input y to the func is ok
def is_stationary(y,regression='ctt'):
	# first import the adfuller test module
	from statsmodels.tsa.stattools import adfuller
	#     if window is not None:
	#         if detrend:
	#             message("Ding Dickey-Fuller test with freq: %s, window: %s after detrend" %(str(freq),str(window)))
	#         else:
	#             message("Ding Dickey-Fuller test with freq: %s, window: %s after smoothed" %(str(freq),str(window)))
	#     else:
	#         message("Ding Dickey-Fuller test with raw data")

	message('Results of Dickey-Fuller Test: ')
	dftest = adfuller(y, regression=regression,autolag = 'AIC')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key, value in dftest[4].items():
		dfoutput['Critical Value (%s)' %key] = value
	print (dfoutput)
	if dfoutput['p-value'] > 0.05:
		message('null hypothesis are failed to be rejected,the timeseries is non-stationary')
		return False
	else:
		message('null hypothesis are rejected, the timeseries is stationary ')
		return True

# test if two time series are cointegration, like MAT and PAT
## in fact, we first calculate the hedge ratio by running a regression on one series 
## against the other, we get the beta(also called hedge ratio); then we take the diff
## between the two time series with the hedge ratio, then do the is_stationary test
def is_cointegration(y1,y2,trend='ct'):
	# first import the adfuller test module
	from statsmodels.tsa.stattools import coint
	message('Results of Engle-Granger Test: ')
	cointtest = coint(y1,y2,trend=trend, autolag = "AIC")
	cointoutput = pd.Series(cointtest[0:2], index=['coint_t','p-value'])
	#     for key, value in cointtest[2].items():
	#         cointoutput['Critical Value (%s)' %key] = value
	key_dict = {0:'1\%',1:'5\%',2:'10\%'}
	for i, value in enumerate(cointtest[2]):
		cointoutput['Critical Value (%s)' %key_dict[i]] = value
# 	print (cointoutput)
	if cointoutput['p-value'] > 0.05:
		message('null hypothesis are failed to be rejected,the timeseries is non-coint')
		return False
	else:
		message('null hypothesis are rejected, the timeseries is coint ')
		return True

# get time range union
## get time union in a sol, now just for two code union
def time_union(sol,code=["MAT","PAT"],mint=None,maxt=None,timetype='LTST',npyfolder=None,kind='linear'):
	# the len of code should be 2
	data = [np.ndarray(shape=1), np.ndarray(shape=1)]
	time = [np.ndarray(shape=1), np.ndarray(shape=1)]
	#     field = [np.ndarray(shape=1), np.ndarray(shape=1)]
	xmin = [0,0]
	xmax = [0,0]

	for i,var_name in enumerate(code):
		if var_name == "PRE":
			data_bundle = 'ps'
		elif var_name in ["HWS","WD","MAT","PAT"]:
			data_bundle = 'twins'
		data[i] = getsol(sol,npyfolder,data_bundle)
		time[i] = gettime(data[i],timetype)
		w =getwhere(time[i],mint=mint,maxt=maxt)
		time[i] = time[i][w]
		xmin, xmax = np.min(time[i]), np.max(time[i])
		xmin, xmax = np.max([mint,xmin]), np.min([maxt,xmax])
		data[i] = data[i][var_name][w]
		ww = np.isfinite(data[i])
		data[i] = data[i][ww]
		time[i] = time[i][ww]

	xmin_all = np.min(xmin)
	xmax_all = np.max(xmax)
	if len(time[0][(time[0]>= xmin_all) & (time[0] <= xmax_all)]) > len(time[1][(time[1]>= xmin_all) & (time[1] <= xmax_all)]):
		time_all = time[0][(time[0]>= xmin_all) & (time[0] <= xmax_all)]
		data[0] = data[0][(time[0]>= xmin_all) & (time[0] <= xmax_all)]
		func = spint.interpolate.interp1d(time[1], data[1], kind=kind, fill_value="extrapolate")
		data[1] = func(time_all)
	else:
		time_all = time[1][(time[1]>= xmin_all) & (time[1] <= xmax_all)]
		data[1] = data[1][(time[1]>= xmin_all) & (time[1] <= xmax_all)]
		func = spint.interpolate.interp1d(time[0], data[0], kind=kind, fill_value="extrapolate")
		data[0] = func(time_all)

	return time_all, data

# get data in a sol range
def getdatamultisol(solini,solsol,code="MAT",mint=None,maxt=None,timetype='LTST',npyfolder=None, \
					freq=10,window=2000,addsmooth=False,):
	'''
	solini: starting sol 
	solsol: sol range
	'''
	if code == "PRE":
		data_bundle = 'ps'
	elif code in ["HWS","WD","MAT","PAT"]:
		data_bundle = 'twins'

	# 
	x = np.array([])
	y = np.array([])
	xref = np.array([])
	solref = np.array([])

	for sol in tqdm(range(solini,solini+solsol+1)):
		try:
			data = getsol(sol,npyfolder,data_bundle)
			time = gettime(data,timetype)

			### since the rmdisorder can lead to unsuitable slice, we add these code here
			# TBD: wrap to a function
			# one point at the beginning might not be exactly midnight
			# if time[0] > 23.9999:
			# 	time[0] = 0.
			# a couple points are the next day in the end
			# correct this to add 24 to thost LMST
			# mind to know that only works for complete sols
			w = np.where(np.diff(time) < 0)
			if len(w) > 0:
				w = np.append(w, len(time) - 1)
				w = np.insert(w, 0, 0)
				time_range = np.diff(w).tolist()
				idx = time_range.index(max(time_range))
				# time[w[idx]+1:w[idx+1]] = time[w[idx]+1:w[idx+1]] +24.
				time[:w[idx] + 1] += 24
				time[w[idx + 1]:] += 24
			### since the rmdisorder can lead to unsuitable slice, we add these code here

			timeref = data['SCLK']
			if freq is None:
				freq = np.int64(np.round(1./getmode(np.diff(timeref)))) # get the freq from the time series
			### if window is provided, smooth and detrend
			if window is not None:
				dpp, spp = smoothresample(data,code,freq=freq,window=window,complete=True)

			if addsmooth:
				data = spp
			else:
				data = dpp
			w = getwhere(time,mint=mint,maxt=maxt)
			time = time[w]
			# 		timeref = data['SCLK'][w]
			# 		data = data[code][w]
			timeref = timeref[w]
			data = data[w]
			ww = np.isfinite(data)
			data = data[ww]
			time = time[ww]
			timeref = timeref[ww] # the ref time for spectrogram/wavelet analysis
			soltime = np.zeros_like(time)+sol
			### actually, that should done just after the getsol function
			# before we append the data, we should first remove all of the disorder timestamp
			# timeref, time, data,soltime = rmdisorder(timeref, time, data, soltime) # do not do it out of the loop in the main func
			###
			x = np.append(x,time)
			xref = np.append(xref,timeref)
			y = np.append(y,data)
			solref = np.append(solref,soltime)
		except:
			message('error in reading npy file of sol %i' %(sol))
			pass
	return y,x,xref,solref #



# since the data bundle are not evenly sampling (some data are detected)
## this is just the sample version of scipy, now we use astropy lomb
def lombscargle(x,y,freqs,normalize=True):
	'''
	x: sample times
	y: Measurement 
	freqs: frequencies for output periodogram unit in HZ
	'''
	freqs = 2*np.pi*freqs # change unit to Angular frequencies
	pgram = signal.lombscargle(x,y,freqs,normalize=normalize)
	return pgram


# moving average
## mind that the output length not the same as the input array
def moving_avg(x,w,mode='valid'):
	# moving average using convolve
	return np.convolve(x,np.ones(w),mode=mode) / w

# normalize data
def normalize(x):
	return (x-np.nanmean(x)) / np.nanstd(x)

# get mode number from an array
def getmode(x):
	from scipy import stats
	return stats.mode(x)[0][0]

# # remove Time disorder
# ### this function is unusable now
# def rmdisorder(xref,x,y,solref=None,ridx=False):
# 	#------old remove disorder version--------#
# 	## this version may lead some sol to be unsuitable cut off
# 	diff = np.diff(x)
# 	reverse_list = [idx for idx, element in enumerate(diff) if element<0]
# 	if reverse_list == []:
# 		return xref, x, y
# 	else:
# 		if len(reverse_list) > 1:
# 			reverse_diff = np.diff(reverse_list).tolist()
# 			max_index = reverse_diff.index(max(reverse_diff))
# 		else:
# 			max_index = len(reverse_list)-1
#
# 		# output
# 		if ridx:
# 			return reverse_list[max_index] + 1
# 		else:
# 			if solref is None:
# 					return xref[reverse_list[max_index]+1:], x[reverse_list[max_index]+1:], y[reverse_list[max_index]+1:]
# 				# return xref[reverse_list[-1]+1:],x[reverse_list[-1]+1:],y[reverse_list[-1]+1:]
# 				# return xref, x[reverse_list[-1]+1:]+24., y # change to the new version
# 			else:
# 				return xref[reverse_list[max_index]+1:], x[reverse_list[max_index]+1:], y[reverse_list[max_index]+1:], \
# 						solref[reverse_list[max_index]+1:]
# 			# return xref[reverse_list[-1]+1:],x[reverse_list[-1]+1:],y[reverse_list[-1]+1:],\
# 			# 	solref[reverse_list[-1]+1:]
# 			# return xref, x[reverse_list[-1]+1:]+24., y, solref # change to the new version
# 	#------old remove disorder version--------#


# butter bandpass filter
## It is a standard filter for continuous signal, but the insight lander signal has a lot of gaps
## we left the filter as an reference
## mind to tune the order parameter!
def butter_band_filter(x,lowcut,highcut,fs,order=2):
	nyq = fs / 2
	low = lowcut / nyq
	high = highcut / nyq

	## the stardand b,a / transfer function representation
	## can bring numerical error
	b, a = signal.butter(order,[low, high], btype='band')

	return signal.lfilter(b,a,x) # return signal after bandpass


# butter bandpass filter dealing with gaps during the sampling
## by befault we use order == 3
## tune it !
## for lowpass order==5 can be ok
## but for highpass just use order==3 or just 2
### mind that the lowcut and highcut unit in s, which is the period not the frequency in Hz!
def butter_band_filter2(xref,y,lowcut=100,highcut=5000,order=3):
	freq = np.int64(np.round(1. / getmode(np.diff(xref))))

	bl, al = signal.butter(order, (1/lowcut) / (freq / 2), 'lowpass') # lowpass butter
	bh, ah = signal.butter(order, (1/highcut) / (freq / 2), 'highpass') # highpass butter

	gap_idx = np.where(np.diff(xref) > 2 * getmode(np.diff(xref)))[0] # find the gaps in the data
	idx_start = 0
	filtered = np.array([])
	for i in range(len(gap_idx)):
		filteredt = signal.filtfilt(bl, al, y[idx_start:gap_idx[i] + 1])
		filteredt = signal.filtfilt(bh, ah, filteredt)
		filtered = np.append(filtered, filteredt)
		idx_start = gap_idx[i] + 1

	filteredt = signal.filtfilt(bl,al,y[idx_start:])
	filteredt = signal.filtfilt(bh,ah,filteredt)
	filtered = np.append(filtered,filteredt)

	return filtered

## make the butter band filter more numerical robust
### this version is numerical robust, but a little bit slow
def butter_band_filter3(xref,y,lowcut=100,highcut=5000,order=3):
	# freq = np.int64(np.round(1. / getmode(np.diff(xref))))
	freq = 1./getmode(np.diff(xref)) # no need to use the int freq
	sos = signal.butter(order,[(1/highcut)/ (freq / 2),(1/lowcut) / (freq / 2)],btype='band',output='sos')
	gap_idx = np.where(np.diff(xref) > 2 * getmode(np.diff(xref)))[0]  # find the gaps in the data
	idx_start = 0
	filtered = np.array([])
	for i in range(len(gap_idx)):
		### The function sosfiltfilt from scipy.signal is a forward-backward filter
		### it applies the filter twice, once forward and one backward. This effectively
		### doubles the order of the filter and the results in zero phase shift
		filteredt = signal.sosfiltfilt(sos,y[idx_start:gap_idx[i] + 1])
		filtered = np.append(filtered,filteredt)
		idx_start = gap_idx[i] + 1
	filteredt = signal.sosfiltfilt(sos,y[idx_start:])
	filtered = np.append(filtered,filteredt)
	return filtered

## because we just wanna find top n frequency with highest psd
## so we can just use the heap, not to sort the list which is costly.
def find_n_max(x,max_n):
	import heapq
	return list(map(x.index, heapq.nlargest(max_n,x))) # mind that the map() return a map object, which should be change to list object for slicing

# get the spectrum of data
# show the spectrum in a sol
def spectrum_analysis(x,y,sampling,title,max_n=5,confidence=False):
	## now we just use FFT method
	from spectrum import speriodogram
	from scipy.stats import gamma  # for normal case, just use the gamma distribution
	# from scipy.stats import chi2 # import the chi2 square distribution

	plt.figure(figsize=(15,10))
	psd = speriodogram(y,detrend=True,sampling=sampling,scale_by_freq=True,window='hamming')
	# because the NFFT is not the same with the sampling, we need to rescale it
	frequency = np.arange(0, len(psd), 1) / (len(y) / sampling)

	#-----calculate the confidence level------#
	threshold = np.percentile(psd,95) # default for 95% confidence level
	filtered = [x for x in psd if x<= threshold] # remove the outliers
	var = np.mean(filtered) # estimate variance
	alpha = 0.7 # the alpha value for the red noise corr
	rnpsd = (1-alpha**2) / (1+alpha**2-2*alpha*np.cos(2*np.pi*frequency)) # psd of red noise

	# # Monte Carlo Method, unused for now
	# mttimes = 2000
	# psdsig = np.zeros_like(psd)
	# for i in range(mttimes):
	# 	ymt = np.copy(y)
	# 	np.random.shuffle(ymt)
	# 	psdmt = speriodogram(ymt, detrend=False, sampling=sampling, scale_by_freq=True, window='hamming')
	# 	psdsig[psdmt<psd] += 1
	# psdsig = psdsig/mttimes
	# psdsig[psdsig>0.99] = 1


	# -----calculate the confidence level------#

	# plot the result with psd normalized
	plt.plot(1. / frequency, psd * var / len(filtered), marker='o')
	plt.plot(1. / frequency, rnpsd * gamma.isf(q=0.05, a=1, scale=var), 'r--')
	plt.xscale('log')
	plt.yscale('log')

	plt.grid(True)
	plt.xlabel('Periods (s)')
	plt.xlim(100, 10 * 10 ** 3)
	plt.ylim(np.min(psd * var / len(filtered)), np.max(psd * var / len(filtered)))
	plt.ylabel('PSD')
	plt.title(title)
	plt.savefig('./img/' + title + '.png', dpi=600, format='png', bbox_inches='tight')
	plt.close()

	# get the max n frequency with the highest psd
	# TBD: maybe we should also add the pds info

	# ## check for the Nyquist–Shannon sampling theorem with the top n psd, not use now
	# freqs = [freq for freq in frequency[find_n_max(psd.tolist(), max_n)] if freq <= (sampling/2)]

	## filter the frequency above the psd confidence level
	freqs = frequency[(psd * var / len(filtered)) > (rnpsd*gamma.isf(q=0.05,a=1,scale=var))]
	scaled_psd = (psd * var / len(filtered))[(psd * var / len(filtered)) > (rnpsd*gamma.isf(q=0.05,a=1,scale=var))]
	return freqs, scaled_psd

# remove the tidal signal
## mind to know input the original not the detrend signal to the y parameter
def remove_tides(x,y):
	import scipy
	tides_periods = np.array([8, 12, 24]) * 3600
	tides_freqs = (2 * np.pi) * np.ones(tides_periods.shape) / tides_periods

	# define a set of tide modes
	def cos_func1(x,A,p,c):
		return A*np.cos(tides_freqs[0]*x+p)+c

	def cos_func2(x,A,p,c):
		return A*np.cos(tides_freqs[1]*x+p)+c

	def cos_func3(x,A,p,c):
		return A*np.cos(tides_freqs[2]*x+p)+c

	# optimize to find the best curve, mind to the set the initial guess
	popt, pcov = scipy.optimize.curve_fit(cos_func1,x,y,method='dogbox',maxfev=2000,p0=[np.std(y) * 2.**0.5,0,np.mean(y)])
	A,p,c = popt
	fitfunc1 = lambda x: A*np.cos(tides_freqs[0]*x+p)+c
	ym1 = fitfunc1(x)

	popt, pcov = scipy.optimize.curve_fit(cos_func2, x, y-ym1, method='dogbox', maxfev=2000,\
	                                      p0=[np.std(y - ym1) * 2. ** 0.5, 0, np.mean(y - ym1)])
	A, p, c = popt
	fitfunc2 = lambda x: A*np.cos(tides_freqs[1]*x+p)+c
	ym2 = fitfunc2(x)

	popt, pcov = scipy.optimize.curve_fit(cos_func3, x, y-ym1-ym2, method='dogbox', maxfev=2000,\
	                                      p0=[np.std(y - ym1 - ym2) * 2. ** 0.5, 0, np.mean(y - ym1 - ym2)])
	A, p, c = popt
	fitfunc3 = lambda x: A*np.cos(tides_freqs[1]*x+p)+c
	ym3 = fitfunc3(x)

	return y - ym1 - ym2 - ym3 # return the y data after removing tides signals

# get the nearest position in a array
def get_nearpos(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

# harmonic fitting with SVD
# the phase == True version is not robust!!!
def harmonic_fit(x,y,xf=None,order=8,Tperiod=24,phase=False,returnp=False):
	ym = np.nanmean(y)
	trifuncs = []
	for k in range(1,order+1):
		if phase:
			trifuncs.append(np.e**(np.e**np.complex(1j)*2*np.pi*k/Tperiod*x))
			trifuncs.append(np.e**(np.e**np.complex(-1j)*2*np.pi*k/Tperiod*x))
		else:
			trifuncs.append(np.cos(2*np.pi*k/Tperiod*x).tolist())
			trifuncs.append(np.sin(2*np.pi*k/Tperiod*x).tolist())
	trimatrix = np.mat(trifuncs).T 
	U, sigma, VT = np.linalg.svd(trimatrix,full_matrices=False)
	V = VT.T
	ST = np.diag(1/sigma)
	B = np.array(V.dot(ST).dot(U.T).dot(y-ym))[0,:]
	
	if phase:
		amp = np.zeros(len(B)//2) 
		phi = np.zeros(len(B)//2) # real: cos part, imag: sin part
		for i  in range(0,len(B),2):
			amp[i//2] = np.sqrt(B[i].real**2+B[i+1].imag**2)
			phi[i//2] = np.arctan(B[i+1].imag/B[i].real)
	else:
		amp = B
	
	# begin fit with xf
	if xf is not None:
		trifuncs = []
		for k in range(1,order+1):
			if phase:
				trifuncs.append(np.cos(2*np.pi*k/Tperiod*xf+phi[k-1]).tolist())
			else:
				trifuncs.append(np.cos(2*np.pi*k/Tperiod*xf).tolist())
				trifuncs.append(np.sin(2*np.pi*k/Tperiod*xf).tolist())
		trimatrix = np.mat(trifuncs).T 
		yrecon = np.squeeze(np.array([amp[i]*trimatrix[:,i] for i in range(len(amp))]))
		yrecon = ym + np.nansum(yrecon,axis=0)
	
	if returnp:
		if phase:
			return yrecon, amp, phi
		else:
			return yrecon, amp
	else:
		return yrecon