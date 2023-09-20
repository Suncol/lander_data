from LanderData import LanderData


# Main function to call LanderData class
if __name__ == '__main__': 

    # # for curiosity lander
    # lander_name = 'msl'
    # data_path = '/storage/aemolcore02/suncong/observations/landers/msl/DATA'
    # msl_obj = LanderData(lander_name, data_path)
    # # msl_obj._get_cached()
    # msl_obj.run()

    # for insight lander
    lander_name = 'insight'
    # ps_data_path = '/storage/aemolcore02/suncong/observations/landers/insight/ps_bundle/data_calibrated'
    twins_data_path = 'D:\\data\\landers_data\\twins_bundle\\data_derived'
    # insight_obj = LanderData(lander_name, ps_data_path)
    # insight_obj.run()
    # insight_obj._get_cached()
    insight_obj = LanderData(lander_name, twins_data_path, warm_start=True)
    # insight_obj._get_cached()
    insight_obj.run()
    #
    # # for mars2020 lander
    # lander_name = 'mars2020'
    # calib_data_path = '/storage/aemolcore02/suncong/observations/landers/mars2020/mars2020_meda/data_calibrated_env'
    # derived_data_path = '/storage/aemolcore02/suncong/observations/landers/mars2020/mars2020_meda/data_derived_env'
    # mars2020_obj = LanderData(lander_name, calib_data_path)
    # mars2020_obj.run()
    # # mars2020_obj._get_cached()
    # # mars2020_obj = LanderData(lander_name, derived_data_path)
    # # mars2020_obj._get_cached()