import numpy as np
import re
from tqdm import tqdm
import os
import insightp

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


# define IO code for the mars 2020 data bundle
mv = -9999





dict_name = {'PRESSURE':'PRESSURE','HORIZONTAL_WIND_SPEED':'HORIZONTAL_WIND_SPEED',\
             'VERTICAL_WIND_SPEED':'VERTICAL_WIND_SPEED','WIND_DIRECTION':'WIND_DIRECTION',\
            'ATS_LOCAL_TEMP1':'ATS_LOCAL_TEMP1','ATS_LOCAL_TEMP2':'ATS_LOCAL_TEMP2','ATS_LOCAL_TEMP3':'ATS_LOCAL_TEMP3',\
            'ATS_LOCAL_TEMP4':'ATS_LOCAL_TEMP4','ATS_LOCAL_TEMP5':'ATS_LOCAL_TEMP5'}

dict_unit = {'PRESSURE':'Pa','HORIZONTAL_WIND_SPEED':'m/s',\
            'VERTICAL_WIND_SPEED':'m/s','WIND_DIRECTION':'deg','ATS_LOCAL_TEMP1':'K','ATS_LOCAL_TEMP2':'K',\
            'ATS_LOCAL_TEMP3':'K','ATS_LOCAL_TEMP4':'K','ATS_LOCAL_TEMP5':'K'}

def mars2020_sol2ls(soltabin,forcecontinuity=True):
	solzero = 13
	return insightp.mars_sol2ls(soltabin + solzero, forcecontinuity=forcecontinuity)

def dir_name_sort(dir_name_list,des=False):
	dir_name_list.sort(key=lambda x: int(x.split('_')[-1]),reverse=des)
	return dir_name_list

def get_data_dirs(data_path):
	dirs = [data_dir for data_dir in os.listdir(data_path) if os.path.os.path.isdir(os.path.join(data_path,data_dir))]
	reg = re.compile(r'sol_\d{4}')
	return dir_name_sort(list(filter(reg.match,dirs)))

def csv2npy(data_path,data_dirs,npyfolder,data_type='PS'): 
    # we prefer to load derived data bundle
    if data_type == "PS": # if load the pressure data bundle
        dtype = [('SCLK', '<f8'),('LMST','S18'),('LTST', 'S13'),\
            ('PRESSURE','<f8'),('PRESSURE_UNCERTAINTY','<i8'),\
                ('TRANSDUCER','<i8')] # since we do not know the PRESSURE_UNCERTAINTY type, use integer here
        reg = re.compile(r'WE__\d{4}___________DER_PS__________________P\d{2}.CSV')
    elif data_type == "WS": # if load the wind data bundle
        dtype = [('SCLK', '<f8'),('LMST','S18'),('LTST', 'S13'),\
                 ('HORIZONTAL_WIND_SPEED','<f8'),('HORIZONTAL_WIND_SPEED_UNCERTAINTY','<f8'),\
                 ('VERTICAL_WIND_SPEED','<f8'),('VERTICAL_WIND_SPEED_UNCERTAINTY','<f8'),\
                 ('WIND_DIRECTION','<f8'),('WIND_DIRECTION_UNCERTAINTY','<f8'),\
                 ('BOTH_BOOMS_USED_FOR_RETRIEVAL','<i8'),('ROVER_STILL','<i8')]
        reg = re.compile(r'WE__\d{4}___________DER_WS__________________P\d{2}.CSV')
    elif data_type in ["ATS_LOCAL_TEMP1","ATS_LOCAL_TEMP2","ATS_LOCAL_TEMP3","ATS_LOCAL_TEMP4","ATS_LOCAL_TEMP5"]:
        dtype = [('SCLK', '<f8'),('LMST','S18'),('LTST', 'S13'),\
                 ('ATS_LOCAL_TEMP1','<f8'),('ATS_LOCAL_TEMP2','<f8'),('ATS_LOCAL_TEMP3','<f8'),\
                 ('ATS_LOCAL_TEMP4','<f8'),('ATS_LOCAL_TEMP5','<f8')]
        reg = re.compile(r'WE__\d{4}___________CAL_ATS_________________P\d{2}.CSV')
    elif data_type in ["DOWNWARD_LW_IRRADIANCE","DOWNWARD_LW_IRRADIANCE_UNCERTAINTY","AIR_TEMP","AIR_TEMP_UNCERTAINTY",\
                       "UPWARD_SW_IRRADIANCE","UPWARD_SW_IRRADIANCE_UNCERTAINTY","UPWARD_LW_IRRADIANCE","UPWARD_LW_IRRADIANCE_UNCERTAINTY",\
                        "GROUND_TEMP","GROUND_TEMP_UNCERTAINTY"]:
        dtype = [('SCLK', '<f8'),('LMST','S18'),('LTST', 'S13'),('DOWNWARD_LW_IRRADIANCE','<f8'),('DOWNWARD_LW_IRRADIANCE_UNCERTAINTY','<f8'),\
                 ('AIR_TEMP','<f8'),('AIR_TEMP_UNCERTAINTY','<f8'),('UPWARD_SW_IRRADIANCE','<f8'),('UPWARD_SW_IRRADIANCE_UNCERTAINTY','<f8'),\
                 ('UPWARD_LW_IRRADIANCE','<f8'),('UPWARD_LW_IRRADIANCE_UNCERTAINTY','<f8'),('GROUND_TEMP','<f8'),('GROUND_TEMP_UNCERTAINTY','<f8'),\
                    ('RSM_HEAD_OUTSIDE_TIRS_UPWARD_LOOKING_FOV','<i8'),("WHEEL_OUTSIDE_TIRS_DOWNWARD_LOOKING_FOV",'<i8'),\
                    ('SUN_OUTSIDE_TIRS_FOV','<i8'),('ROVER_LOW_TILT','<i8'),('TIRS_GROUND_FOOTPRINT_NOT_IN_SHADOW','<i8'),('ROVER_HGA_OFF','<i8'),\
                       ('SKYCAM_OFF','<i8'),('ROVER_STILL','<i8') ]
        reg = re.compile(r'WE__\d{4}___________CAL_TIRS________________P\d{2}.CSV')
    else: 
        dtype = None # need to be checked later

    from joblib import Parallel, delayed
    def save_to_csvfile(sol_dir):
        if os.path.isdir(os.path.join(data_path,adir,sol_dir)):
            file_paths = [file_path for file_path in \
                os.listdir(os.path.join(data_path,adir,sol_dir)) \
                    if os.path.isfile(os.path.join(data_path,adir,sol_dir,file_path))]
            try:
                file_name = list(filter(reg.match,file_paths))[-1]
            except:
                print(file_paths)
                print(' File is not found')
                return 
            npy_file = file_name[:-3]+'npy'
            if not os.path.exists(os.path.join(npyfolder,npy_file)):
                try: 
                    data = np.genfromtxt(os.path.join(data_path,adir,sol_dir,file_name),\
                        dtype=dtype,names=True,delimiter=',',filling_values=(mv))
                except:
                    print(' File is not readable')
                    data = None
                    np.save(os.path.join(npyfolder,npy_file),data)
                    print("saved a VOID Numpy binary file: " + npy_file)
                    #return
                for name in data.dtype.names:
                    try:
                        w = np.where(data[name]==-9999)
                        data[name][w] = np.nan
                    except:
                        pass
                np.save(os.path.join(npyfolder,npy_file),data)


    for adir in tqdm(data_dirs,desc='loop in data_dirs'):
        sol_dirs = os.listdir(os.path.join(data_path,adir))
        Parallel(n_jobs=n_jobs)(delayed(save_to_csvfile)(sol_dir) for sol_dir in sol_dirs)
        # for sol_dir in sol_dirs:
        #     if os.path.isdir(os.path.join(data_path,adir,sol_dir)):
        #         file_paths = [file_path for file_path in \
        #             os.listdir(os.path.join(data_path,adir,sol_dir)) \
        #                 if os.path.isfile(os.path.join(data_path,adir,sol_dir,file_path))]
        #         try:
        #             file_name = list(filter(reg.match,file_paths))[-1]
        #         except:
        #             print(file_paths)
        #             print(' File is not found')
        #             continue
        #         npy_file = file_name[:-3]+'npy'
        #         if not os.path.exists(os.path.join(npyfolder,npy_file)):
        #             try: 
        #                 data = np.genfromtxt(os.path.join(data_path,adir,sol_dir,file_name),\
        #                     dtype=dtype,names=True,delimiter=',',filling_values=(mv))
        #             except:
        #                 print(' File is not readable')
        #                 data = None
        #                 np.save(os.path.join(npyfolder,npy_file),data)
        #                 print("saved a VOID Numpy binary file: " + npy_file)
        #                 #return
        #             for name in data.dtype.names:
        #                 try:
        #                     w = np.where(data[name]==-9999)
        #                     data[name][w] = np.nan
        #                 except:
        #                     pass
        #             np.save(os.path.join(npyfolder,npy_file),data)

def gettime(data,timetype):
    time = data[timetype]
    # convert to float
    if timetype == 'LTST':
        time = insightp.ltstfloat(time, indices=[5, 7, 8, 10, 11, 13])
    elif timetype == 'LMST':
        time = insightp.lmstfloat(time)
    return time

# get data in a sol, mind to preprocess the data file to npy
def getsol(sol,npyfolder,data_type='PS'):
    if data_type == "PS":
        reg = re.compile(r"%s" \
            %('WE__%s___________DER_PS__________________P\d{2}.npy')\
            %('{0:0>4}'.format(sol)))
    elif data_type == "HORIZONTAL_WIND_SPEED" or data_type == "WIND_DIRECTION" or data_type == "VERTICAL_WIND_SPEED":
        reg = re.compile(r"%s" \
            %('WE__%s___________DER_WS__________________P\d{2}.npy')\
            %('{0:0>4}'.format(sol)))
    elif data_type in ["ATS_LOCAL_TEMP1","ATS_LOCAL_TEMP2","ATS_LOCAL_TEMP3","ATS_LOCAL_TEMP4","ATS_LOCAL_TEMP5"]:
        reg = re.compile(r"%s" \
            %('WE__%s___________CAL_ATS_________________P\d{2}.npy')\
            %('{0:0>4}'.format(sol)))
    elif data_type in ["DOWNWARD_LW_IRRADIANCE","DOWNWARD_LW_IRRADIANCE_UNCERTAINTY","AIR_TEMP","AIR_TEMP_UNCERTAINTY",\
                       "UPWARD_SW_IRRADIANCE","UPWARD_SW_IRRADIANCE_UNCERTAINTY","UPWARD_LW_IRRADIANCE","UPWARD_LW_IRRADIANCE_UNCERTAINTY",\
                        "GROUND_TEMP","GROUND_TEMP_UNCERTAINTY"]:
        reg = re.compile(r"%s" \
            %('WE__%s___________CAL_TIRS________________P\d{2}.npy')\
            %('{0:0>4}'.format(sol)))
    else:
        reg = None
        print('using unknown data type!')
    npyfiles = [f for f in os.listdir(npyfolder) \
        if os.path.isfile(os.path.join(npyfolder, f))]
    sol_files = list(filter(reg.match,npyfiles))
    return np.load(os.path.join(npyfolder,sol_files[0]))

def getinsol(sol,code='PRESSURE',mint=None,maxt=None,\
    timetype='LMST',freq=1.,window=None,addsmooth=False,\
        npyfolder=None):
    # print('getinsol: ------------------------%.2f' %freq)
    if code == 'PRESSURE':
        data_type = 'PS'
    elif code == 'HORIZONTAL_WIND_SPEED' or code == "WIND_DIRECTION" or code == "VERTICAL_WIND_SPEED":
        data_type = code
    elif code in ["ATS_LOCAL_TEMP1","ATS_LOCAL_TEMP2","ATS_LOCAL_TEMP3","ATS_LOCAL_TEMP4","ATS_LOCAL_TEMP5"]:
        data_type = code
    elif code in ["DOWNWARD_LW_IRRADIANCE","DOWNWARD_LW_IRRADIANCE_UNCERTAINTY","AIR_TEMP","AIR_TEMP_UNCERTAINTY",\
                       "UPWARD_SW_IRRADIANCE","UPWARD_SW_IRRADIANCE_UNCERTAINTY","UPWARD_LW_IRRADIANCE","UPWARD_LW_IRRADIANCE_UNCERTAINTY",\
                        "GROUND_TEMP","GROUND_TEMP_UNCERTAINTY"]:
        data_type = code
    else:
        raise('code error with code: %s!' %code)
    # print('data_type: %s' %data_type)
    data = getsol(sol,npyfolder,data_type=data_type)

    # 1. get times
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
    w = insightp.getwhere(time, mint=mint, maxt=maxt)

    ## 3. reduce time
    time = time[w]

    ## 4. min/max
    xmin, xmax = np.min(time), np.max(time)
    xmin, xmax = np.max([mint,xmin]), np.min([maxt,xmax])

    ### select time interval
    # field = data[code][w] # put it in the if branch
    # field = data[w]
    if freq is None:
        freq = np.int64(np.round(1. / insightp.getmode(np.diff(data['SCLK'])))) # get the real freq from the time series
    ### if window is provided, smooth and detrend
    if window is not None:
        dpp, spp = insightp.smoothresample(data, code, freq=freq, window=window, complete=True)
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
    
    if code == 'HORIZONTAL_WIND_SPEED' or 'WIND_DIRECTION':
        y[y>1000] = np.nan # filter nans
    return y, x, xref 
