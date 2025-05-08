import os
import pandas as pd
from spacepy import pycdf


def process_fgm_model(fgm_data_dir, dates=None, satellite='tha'):
    raw_fgm_matrix = pd.DataFrame(columns=['Time', 'Bx', 'By', 'Bz'])

    for file_name in sorted(os.listdir(fgm_data_dir)):
        if file_name.startswith('.'):  
            continue
        if dates and file_name.split('_')[3] not in dates:
            continue

        cdf_file_fgm = pycdf.CDF(f'{fgm_data_dir}/{file_name}')
        
        time_stamp = cdf_file_fgm[f'{satellite}_fgs_time'][:]
        fgm_data = cdf_file_fgm[f'{satellite}_fgs_gsm'][:]

        temp_df = pd.DataFrame({
            'Time': time_stamp,
            'Bx': fgm_data[:, 0],
            'By': fgm_data[:, 1],
            'Bz': fgm_data[:, 2]
        })

        raw_fgm_matrix = pd.concat([raw_fgm_matrix, temp_df], ignore_index=True)

    return raw_fgm_matrix


def process_state_model(state_data_dir, dates=None, satellite='tha'):
    raw_state_matrix = pd.DataFrame(columns=['Time', 'GSM_x', 'GSM_y'])

    for file_name in sorted(os.listdir(state_data_dir)):
        if file_name.startswith('.'):  
            continue

        if dates and file_name.split('_')[3] not in dates:
            continue
        
        cdf_file_state = pycdf.CDF(f'{state_data_dir}/{file_name}')
        
        time_stamp = cdf_file_state[f'{satellite}_state_time'][:]
        pos_data = cdf_file_state[f'{satellite}_pos_gsm'][:]

        temp_df = pd.DataFrame({
            'Time': time_stamp,
            'GSM_x': pos_data[:, 0],
            'GSM_y': pos_data[:, 1],
        }).apply(lambda col: col.astype(float), axis=0)

        raw_state_matrix = pd.concat([raw_state_matrix, temp_df], ignore_index=True)

    return raw_state_matrix


def process_mom_model(mom_data_dir, dates=None, satellite='tha'):
    raw_iplasma_matrix = pd.DataFrame(columns=['Time', 'I_velocity_x', 'I_velocity_y', 'I_velocity_z'])

    for file_name in sorted(os.listdir(mom_data_dir)):
        if file_name.startswith('.'):  
            continue

        if dates and file_name.split('_')[3] not in dates:
            continue

        cdf_file_mom = pycdf.CDF(f'{mom_data_dir}/{file_name}')

        time_stamp_i = cdf_file_mom[f'{satellite}_peim_time'][:]
        i_flux_velocity = cdf_file_mom[f'{satellite}_peim_velocity_gsm'][:]

        if i_flux_velocity.size == 0:
            continue

        temp_i_df = pd.DataFrame({
            'Time': time_stamp_i,
            'I_velocity_x': i_flux_velocity[:, 0],
            'I_velocity_y': i_flux_velocity[:, 1],
            'I_velocity_z': i_flux_velocity[:, 2]
        }).apply(lambda col: col.astype(float), axis=0)

        raw_iplasma_matrix = pd.concat([raw_iplasma_matrix, temp_i_df], ignore_index=True)

    return raw_iplasma_matrix
