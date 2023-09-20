from scipy.io import savemat
import os
import insightp
import mars2020p
import mslp
# from multiprocessing import Pool
from joblib import Parallel, delayed
from rich.console import Console
from tqdm import tqdm
import socket
import configparser

class LanderData:
    def __init__(self,lander_name,raw_data_path,save_data_path=None,cache_folder=None,fresh_cache=False,warm_start=False):
        self.console = Console()
        n_jobs = None
        hostname = socket.gethostname()
        config =configparser.ConfigParser()
        if save_data_path is None and cache_folder is None:
            try:
                current_directory = os.path.dirname(os.path.abspath(__file__))
                config.read(os.path.join(current_directory, '..', 'config.ini'))
                try:
                    save_data_path = os.path.join(config[hostname]['save_folder'],lander_name)
                    cache_folder = config[hostname]['cache_folder']
                    n_jobs = int(config.get(hostname,'njobs'))
                except:
                    hostname = 'DEFAULT'
                    self.console.print('Hostname not found in config.ini. Trying to use DEFAULT config...',style='bold green')
                    save_data_path = os.path.join(config[hostname]['save_folder'],lander_name)
                    cache_folder = config[hostname]['cache_folder']
                    n_jobs = int(config.get(hostname,'njobs'))
            except:
                self.console.print('The config.ini file does not exist. Please specify the save_data_path and cache_folder in config.ini or in the class.', style='bold red')
                raise ValueError('Please specify the save_data_path and cache_folder in config.ini or in the class.')
        
 
        self.lander_name = lander_name
        self._check_lander_name()
        self.raw_data_path = raw_data_path
        self._check_raw_data_path()
        self.save_data_path = save_data_path
        self._check_save_data_path()
        self.cache_folder = os.path.join(cache_folder,self.lander_name)
        self.fresh_cache = fresh_cache
        self._make_cache_folder()
        self.code_list = None
        if n_jobs is None:
            n_jobs = 1
        self.n_jobs = n_jobs
        self.warm_start = warm_start


    def _get_code_list(self): #### get code list
        if self.lander_name == 'mars2020':
            self.code_list = ['PRESSURE','HORIZONTAL_WIND_SPEED','WIND_DIRECTION','ATS_LOCAL_TEMP1',\
                              "ATS_LOCAL_TEMP2","ATS_LOCAL_TEMP3","ATS_LOCAL_TEMP4","ATS_LOCAL_TEMP5",\
                                "DOWNWARD_LW_IRRADIANCE","DOWNWARD_LW_IRRADIANCE_UNCERTAINTY","AIR_TEMP","AIR_TEMP_UNCERTAINTY",\
                       "UPWARD_SW_IRRADIANCE","UPWARD_SW_IRRADIANCE_UNCERTAINTY","UPWARD_LW_IRRADIANCE","UPWARD_LW_IRRADIANCE_UNCERTAINTY",\
                        "GROUND_TEMP","GROUND_TEMP_UNCERTAINTY"]
        if self.lander_name == 'insight':
            self.code_list = ['PRE', 'PRESSURE_FREQUENCY', \
										'PRESSURE_TEMP', 'PRESSURE_TEMP_FREQUENCY','HWS', 'VERTICAL_WIND_SPEED', \
										'WD', 'WIND_FREQUENCY', 'WS_OPERATIONAL_FLAGS', \
										'MAT', 'BMY_AIR_TEMP_FREQUENCY', 'BMY_AIR_TEMP_OPERATIONAL_FLAGS', \
										'PAT', 'BPY_AIR_TEMP_FREQUENCY', 'BPY_AIR_TEMP_OPERATIONAL_FLAGS']
        if self.lander_name == 'msl':
            self.code_list = ['PRESSURE', 'HORIZONTAL_WIND_SPEED', 'VERTICAL_WIND_SPEED', 'WIND_DIRECTION', \
                              'BRIGHTNESS_TEMP', 'BOOM1_LOCAL_AIR_TEMP', 'BOOM2_LOCAL_AIR_TEMP', 'AMBIENT_TEMP', \
                                'UV_A', 'UV_B', 'UV_C', 'UV_ABC', 'UV_D', 'UV_E', 'HS_TEMP', 'VOLUME_MIXING_RATIO']
    
    def _check_lander_name(self):
        avail_lander = ['insight','msl','mars2020','zhurong']
        if self.lander_name not in avail_lander:
            self.console.print('Lander name not available. Please choose from [bold green]{}[/bold green]'.format(avail_lander))
            raise ValueError('Lander name not available.')
        self.console.print('Lander name: [bold green]{}[/bold green]'.format(self.lander_name))        
    
    def _check_raw_data_path(self):
        if not self.check_path(self.raw_data_path):
            raise ValueError('Data path: %s does not exist.' %self.raw_data_path)
    
    def _check_save_data_path(self):
        if not self.check_path(self.save_data_path):
            self.console.print('Save data path: [bold green]{}[/bold green] does not exist. Creating...'.format(self.save_data_path))
            os.makedirs(self.save_data_path)
        else:
            self.console.print('Save data path: [bold green]{}[/bold green]'.format(self.save_data_path))
            
    
    @staticmethod
    def check_path(data_path):
        return os.path.exists(data_path)
    
    def _make_cache_folder(self):
        if not self.check_path(self.cache_folder):
            self.console.print('Making cache folder in %s ...' %self.cache_folder, style='bold green')
            os.makedirs(self.cache_folder)
        else:
            self.console.print('Cache folder already exists in %s' %self.cache_folder, style='bold green')
            if self.fresh_cache:
                self.console.print('Fresh cache is set to [bold green]True[/bold green]. Removing cache folder...')
                os.removedirs(self.cache_folder)
                self.console.print('Old cache file deleted, creating new cache folder...')
                os.makedirs(self.cache_folder)

    @staticmethod
    def help():
        help_console = Console()
        help_console.print('This is a classs for converting raw lander data to files.\n', style='bold green')
        help_console.print('Now available landers: [bold green]insight, msl, mars2020, zhurong[/bold green]\n', style='bold red')


    def _get_cached(self):
        #### get data dirs
        data_dirs = self._get_data_dirs()
        if self.lander_name == 'mars2020':
            data_type_list = ['PS','WS',"ATS_LOCAL_TEMP1",\
                              "ATS_LOCAL_TEMP2","ATS_LOCAL_TEMP3","ATS_LOCAL_TEMP4","ATS_LOCAL_TEMP5",\
                                "DOWNWARD_LW_IRRADIANCE","DOWNWARD_LW_IRRADIANCE_UNCERTAINTY","AIR_TEMP","AIR_TEMP_UNCERTAINTY",\
                       "UPWARD_SW_IRRADIANCE","UPWARD_SW_IRRADIANCE_UNCERTAINTY","UPWARD_LW_IRRADIANCE","UPWARD_LW_IRRADIANCE_UNCERTAINTY",\
                        "GROUND_TEMP","GROUND_TEMP_UNCERTAINTY"]
            csv2npy_func = mars2020p.csv2npy
        elif self.lander_name == 'insight':
            data_type_list = ['PS','TWINS']
            csv2npy_func = insightp.csv2npy
            pass
        elif self.lander_name == 'msl':
            data_type_list = ['ptw']
            csv2npy_func = mslp.csv2npy
            pass
        elif self.lander_name == 'zhurong':
            pass
            
        # ### get cache files
        for data_type in data_type_list: # mind the different data types
            self.console.print('Processing [bold green]{}[/bold green] data...\n'.format(data_type))
            self.console.print('Converting raw files to npy files for cache...\n')
            # if self.lander_name == 'mars2020' or self.lander_name == 'insight':
            #     csv2npy_func(self.raw_data_path,data_dirs,self.cache_folder,data_type) # skip files that already cached
            # elif self.lander_name == 'msl':
            #     csv2npy_func(self.raw_data_path,self.cache_folder,data_type) 
            self.cal_csv2npy(csv2npy_func,data_dirs,data_type)        

        # Parallel(n_jobs=len(data_type_list))(delayed(self.cal_csv2npy)(csv2npy_func,data_dirs,data_type) for data_type in tqdm(data_type_list))

    def cal_csv2npy(self,csv2npy_func,data_dirs,data_type):
        if self.lander_name == 'mars2020' or self.lander_name == 'insight':
            csv2npy_func(self.raw_data_path,data_dirs,self.cache_folder,data_type) # skip files that already cached
        elif self.lander_name == 'msl':
            csv2npy_func(self.raw_data_path,self.cache_folder,data_type)

            
    def run(self):
        # #### get data dirs
        # data_dirs = self._get_data_dirs()
        # if self.lander_name == 'mars2020':
        #     data_type_list = ['PS','WS']
        #     csv2npy_func = mars2020p.csv2npy
        # elif self.lander_name == 'insight':
        #     data_type_list = ['PS','TWINS']
        #     csv2npy_func = insightp.csv2npy
        #     pass
        # elif self.lander_name == 'msl':
        #     data_type_list = ['ptw']
        #     csv2npy_func = mslp.csv2npy
        #     pass
        # elif self.lander_name == 'zhurong':
        #     pass
            
        # ### get cache files
        # for data_type in data_type_list: # mind the different data types
        #     self.console.print('Processing [bold green]{}[/bold green] data...\n'.format(data_type))
        #     self.console.print('Converting raw files to npy files for cache...\n')
        #     if self.lander_name == 'mars2020' or self.lander_name == 'insight':
        #         csv2npy_func(self.raw_data_path,data_dirs,self.cache_folder,data_type) # skip files that already cached
        #     elif self.lander_name == 'msl':
        #         csv2npy_func(self.raw_data_path,self.cache_folder,data_type) 
        if not self.warm_start:
            self._get_cached() # in case there is not cached files

        ### getinsol files and loop for all  variables
        ### save results of all variables to mat files
        min_sol, max_sol = self._get_sol_range()
        code_list = self._get_code_list()
        self.console.print('Getting %s data between Sol: %04i-%04i\n' %(self.lander_name,min_sol,max_sol))

        for code in self.code_list:
            for time_type in ['LTST','LMST']:
                Parallel(n_jobs=self.n_jobs)(\
                    delayed(self.get_insol_data_saved_para_for)(sol,code,time_type,self.lander_name,self.cache_folder,self.save_data_path) for sol in tqdm(range(min_sol,max_sol+1)))

        # for sol in tqdm(range(min_sol,max_sol+1),desc='loop in Sols'): # maybe do some parallel here!
        #     self._save_data_in_sol(sol)
        # Parallel(n_jobs=self.n_jobs)(delayed(self._save_data_in_sol)(sol) for sol in tqdm(range(min_sol,max_sol+1),desc='loop in Sols'))
        # with Pool(self.n_jobs) as pool:
        #     pool.map(self._save_data_in_sol,list(range(min_sol,max_sol+1)))
        #     tqdm(pool.imap(self._save_data_in_sol,list(range(min_sol,max_sol+1))),total=max_sol-min_sol+1,desc='loop in Sols')

            # for code in self.code_list:
            #     for time_type in ['LTST','LMST']:
            #         try:
            #             data_dict = self.get_insol_data(sol,code,time_type)
            #             self._save_data(data_dict)
            #         except:
            #             pass
            #             # self.console.print('Error when getting data for Sol: %04i, code: %s, time_type: %s' %(sol,code,time_type), style='bold red')
            #             # self.console.print('Maybe there is no data for this sol, code, time_type combination.', style='bold red')


    def _save_data_in_sol(self, sol):
        for code in self.code_list:
            for time_type in ['LTST','LMST']:
                try:
                    data_dict = self.get_insol_data(sol, code, time_type)
                    self._save_data(data_dict)
                except:
                    pass
                    self.console.print('Error when getting data for Sol: %04i, code: %s, time_type: %s' %(sol,code,time_type), style='bold red')
                    self.console.print('Maybe there is no data for this sol, code, time_type combination.', style='bold red')

    def _save_data(self,data_dict,to_file_type='mat'):
        if to_file_type == 'mat':
            file_name = os.path.join(self.save_data_path,'%s_%s_SOL_%04i_timetype_%s.mat'\
                                     %(self.lander_name,data_dict['data_name'],data_dict['sol'],data_dict['timetype']))
            if os.path.exists(file_name):
                # self.console.print('File %s already exists. Skip saving...' %file_name, style='bold green')
                return
            try:
                savemat(file_name,data_dict)
            except:
                pass
                # self.console.print('Error when saving data to %s' %file_name, style='bold red')
    @staticmethod
    def get_insol_data_saved_para_for(sol,code,timetype,lander_name,cache_folder,save_data_path):
        try:
            if lander_name == 'mars2020':
                data_tuple = mars2020p.getinsol(sol, code, mint=0, maxt=24, timetype=timetype, npyfolder=cache_folder)
                Ls = mars2020p.mars2020_sol2ls(sol)
            elif lander_name == 'insight':
                data_tuple = insightp.getinsol(sol, code, mint=0, maxt=24, timetype=timetype, npyfolder=cache_folder)
                Ls = insightp.insight_sol2ls(sol)
            elif lander_name == 'msl':
                data_tuple = mslp.getinsol(sol, code, mint=0, maxt=24, timetype=timetype, npyfolder=cache_folder)
                Ls = mslp.msl_sol2ls(sol)
            data_dict = {code: data_tuple[0],'LST':data_tuple[1],'ref_time':data_tuple[2],'data_name':code,'timetype':timetype,'sol':sol,'Ls':Ls}
            file_name = os.path.join(save_data_path, '%s_%s_SOL_%04i_timetype_%s.mat' \
                                     % (lander_name, data_dict['data_name'], data_dict['sol'], data_dict['timetype']))
            savemat(file_name,data_dict)
        except:
            return
    def get_insol_data(self,sol,code,timetype): # by default, we return the data without in sol slice
        try:
            if self.lander_name == 'mars2020':
                data_tuple = mars2020p.getinsol(sol, code, mint=0, maxt=24, timetype=timetype, npyfolder=self.cache_folder)
                Ls = mars2020p.mars2020_sol2ls(sol)
            elif self.lander_name == 'insight':
                data_tuple = insightp.getinsol(sol, code, mint=0, maxt=24, timetype=timetype, npyfolder=self.cache_folder)
                Ls = insightp.insight_sol2ls(sol)
            elif self.lander_name == 'msl':
                data_tuple = mslp.getinsol(sol, code, mint=0, maxt=24, timetype=timetype, npyfolder=self.cache_folder)
                Ls = mslp.msl_sol2ls(sol)
            return  {code:data_tuple[0],'LST':data_tuple[1],'ref_time':data_tuple[2],'data_name':code,'timetype':timetype,'sol':sol,'Ls':Ls}
        except:
            # self.console.print('Error when getting data for Sol: %04i, code: %s, time_type: %s' %(sol,code,timetype), style='bold red')
            # self.console.print('Check the cache file path: %s' %self.cache_folder, style='bold red')
            return None

    def _get_sol_range(self):
        if self.lander_name == 'mars2020' or self.lander_name == 'insight':
            file_list = os.listdir(self.cache_folder)
            sol_list = []
            for file_name in file_list:
                sol_list.append(int(file_name.split('_')[2]))
        if self.lander_name == 'msl':
            file_list = os.listdir(self.cache_folder)
            sol_list = []
            for file_name in file_list:
                sol_list.append(int(file_name.split('_')[0][3:]))
        min_sol = min(sol_list)
        max_sol = max(sol_list)
        return min_sol,max_sol



    def _get_data_dirs(self):
        self.console.print('Getting data directories...', style='bold green')
        if self.lander_name == 'insight':
            return insightp.get_data_dirs(self.raw_data_path)
        elif self.lander_name == 'mars2020':
            return mars2020p.get_data_dirs(self.raw_data_path)
        elif self.lander_name == 'msl':
            return mslp.get_data_dirs(self.raw_data_path)
        # elif self.lander_name == 'zhurong':
        #     return zhurongp.get_data_dirs(self.raw_data_path)

    


# if __name__ == '__main__':
#     # mars2020_obj = lander_data('mars2020','/work/home/suncong/data/mars2020/download/data_derived_env','./mars2020_retrived_data')
#     # mars2020_obj.run()
#     # mars2020_obj.get_insol_data(616,'PRESSURE','LMST')
#     # insightp_obj = lander_data('insight','/work/home/suncong/data/insight/ps/ps_bundle/data_calibrated','./insight_retrived_data')
#     # insightp_obj.run()
#     msl_obj = LanderData('msl','/storage/aemolcore02/suncong/observations/landers/msl/DATA',)
#     msl_obj._get_cached()
#     # pass
