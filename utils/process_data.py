import os
import numpy as np
import pandas as pd
import datetime as dt
from spacepy import pycdf
from scipy.interpolate import interp1d


def filter_outliers(dataframe, columns):
    filtered_df = dataframe.copy()
    
    for column in columns:
        Q1 = filtered_df[column].quantile(0.25)
        Q3 = filtered_df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_df = filtered_df[(filtered_df[column] >= lower_bound) & 
                                  (filtered_df[column] <= upper_bound)]
    
    return filtered_df


def interpolate_columns(source_df, target_times, columns):
    return {col: interp1d(source_df['Time'], source_df[col], kind='linear', fill_value='extrapolate')(target_times) for col in columns}


def process_fgm(fgm_data_dir):
    raw_fgm_matrix = pd.DataFrame(columns=['Time', 'Bx', 'By', 'Bz'])

    for file_name in sorted(os.listdir(fgm_data_dir)):
        cdf_file_fgm = pycdf.CDF(f'{fgm_data_dir}/{file_name}')
        
        time_stamp = cdf_file_fgm['tha_fgs_time'][:]

        bx = cdf_file_fgm['tha_fgs_gsm'][:][:, 0]
        by = cdf_file_fgm['tha_fgs_gsm'][:][:, 1]
        bz = cdf_file_fgm['tha_fgs_gsm'][:][:, 2]

        temp_df = pd.DataFrame({
            'Time': time_stamp,
            'Bx': bx,
            'By': by,
            'Bz': bz
        }).apply(lambda col: col.astype(float), axis=0)

        raw_fgm_matrix = pd.concat([raw_fgm_matrix, temp_df], ignore_index=True)

    return raw_fgm_matrix


def process_state(state_data_dir):
    raw_state_matrix = pd.DataFrame(columns=['Time', 'GSM_x', 'GSM_y', 'GSM_z'])

    for file_name in sorted(os.listdir(state_data_dir)):
        if file_name.endswith('_v02.cdf'):
            cdf_file_state = pycdf.CDF(f'{state_data_dir}/{file_name}')
            
            time_stamp = cdf_file_state['tha_state_time'][:]

            gsm_x = cdf_file_state['tha_pos_gsm'][:][:, 0]
            gsm_y = cdf_file_state['tha_pos_gsm'][:][:, 1]
            gsm_z = cdf_file_state['tha_pos_gsm'][:][:, 2]

            temp_df = pd.DataFrame({
                'Time': time_stamp,
                'GSM_x': gsm_x,
                'GSM_y': gsm_y,
                'GSM_z': gsm_z
            }).apply(lambda col: col.astype(float), axis=0)

            raw_state_matrix = pd.concat([raw_state_matrix, temp_df], ignore_index=True)

    return raw_state_matrix


def process_sst(sst_data_dir):
    raw_eflux_matrix = pd.DataFrame(columns=['Time', 'E_flux_c5'])
    raw_iflux_matrix = pd.DataFrame(columns=['Time', 'I_flux_c5'])

    for file_name in sorted(os.listdir(sst_data_dir)):
        cdf_file_sst = pycdf.CDF(f'{sst_data_dir}/{file_name}')
        
        time_stamp_e = cdf_file_sst['tha_psef_time'][:]
        e_flux_channel_5 = cdf_file_sst['tha_psef_en_eflux'][:][:, 4]

        time_stamp_i = cdf_file_sst['tha_psif_time'][:]
        i_flux_channel_5 = cdf_file_sst['tha_psif_en_eflux'][:][:, 4]

        temp_e_df = pd.DataFrame({
            'Time': time_stamp_e,
            'E_flux_c5': e_flux_channel_5,
        }).apply(lambda col: col.astype(float), axis=0)

        temp_i_df = pd.DataFrame({
            'Time': time_stamp_i,
            'I_flux_c5': i_flux_channel_5,
        }).apply(lambda col: col.astype(float), axis=0)

        raw_eflux_matrix = pd.concat([raw_eflux_matrix, temp_e_df], ignore_index=True)
        raw_iflux_matrix = pd.concat([raw_iflux_matrix, temp_i_df], ignore_index=True)

    return raw_eflux_matrix, raw_iflux_matrix
    

def process_esa(esa_data_dir):
    raw_eplasma_matrix = pd.DataFrame(columns=['Time', 'E_perp1', 'E_perp2', 'E_perp_avg', 'E_prll', 'E_density', 'E_velocity_xy'])
    raw_iplasma_matrix = pd.DataFrame(columns=['Time', 'I_perp1', 'I_perp2', 'I_perp_avg', 'I_prll', 'I_density', 'I_velocity_xy'])

    for file_name in sorted(os.listdir(esa_data_dir)):
        cdf_file_esa = pycdf.CDF(f'{esa_data_dir}/{file_name}')
        
        time_stamp_e = cdf_file_esa['tha_peef_time'][:]
        e_flux_temps = cdf_file_esa['tha_peef_t3'][:]
        e_flux_density = cdf_file_esa['tha_peef_density'][:]
        e_flux_velocity = cdf_file_esa['tha_peef_velocity_gsm'][:]

        time_stamp_i = cdf_file_esa['tha_peif_time'][:]
        i_flux_temps = cdf_file_esa['tha_peif_t3'][:]
        i_flux_density = cdf_file_esa['tha_peif_density'][:]
        i_flux_velocity = cdf_file_esa['tha_peif_velocity_gsm'][:]

        temp_e_df = pd.DataFrame({
            'Time': time_stamp_e,
            'E_perp1': e_flux_temps[:, 0],
            'E_perp2': e_flux_temps[:, 1],
            'E_perp_avg': (e_flux_temps[:, 0] + e_flux_temps[:, 1]) / 2,
            'E_prll': e_flux_temps[:, 2],
            'E_density': e_flux_density,
            'E_velocity_xy': np.sqrt(e_flux_velocity[:, 0] ** 2 + e_flux_velocity[:, 1] ** 2)
        }).apply(lambda col: col.astype(float), axis=0)

        temp_i_df = pd.DataFrame({
            'Time': time_stamp_i,
            'I_perp1': i_flux_temps[:, 0],
            'I_perp2': i_flux_temps[:, 1],
            'I_perp_avg': (i_flux_temps[:, 0] + i_flux_temps[:, 1]) / 2,
            'I_prll': i_flux_temps[:, 2],
            'I_density': i_flux_density,
            'I_velocity_xy': np.sqrt(i_flux_velocity[:, 0] ** 2 + i_flux_velocity[:, 1] ** 2)
        }).apply(lambda col: col.astype(float), axis=0)

        raw_eplasma_matrix = pd.concat([raw_eplasma_matrix, temp_e_df], ignore_index=True)
        raw_iplasma_matrix = pd.concat([raw_iplasma_matrix, temp_i_df], ignore_index=True)

    return raw_eplasma_matrix, raw_iplasma_matrix


def process_omni(omni_data_dir):
    raw_omni_data = pd.read_csv(omni_data_dir)

    raw_omni_data["Time"] = raw_omni_data.apply(lambda row: dt.datetime(int(row["year"]), 1, 1) + 
                                                dt.timedelta(days=int(row["md"]) - 1, 
                                                             hours=int(row["hour"])), axis=1)
    raw_omni_data["Time"] = raw_omni_data["Time"].apply(lambda x: int(x.replace(tzinfo=dt.timezone.utc).timestamp()))
    raw_omni_data.insert(3, "Time", raw_omni_data.pop("Time"))

    return raw_omni_data

