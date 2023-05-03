import pandas as pd
import numpy as np

def get_ats(row, ats):
    citizen_ats = ats.loc[(ats['CitizenId'] == row['RealId']) & \
            (ats['LendDate'].dt.date <= row['RealDate'])]
    citizen_ats = citizen_ats.sort_values(by='LendDate')
    ats_seq = ','.join([str(elem) for elem in citizen_ats['DevISOClass']])
    if not ats_seq:
        return '0'
    return ats_seq

def get_loan_period(row, ats):
    citizen_ats = ats.loc[(ats['CitizenId'] == row['RealId']) & \
            (ats['LendDate'].dt.date <= row['RealDate'])]
    if citizen_ats.empty != True:
        lend_diff = citizen_ats['LendDate'] - citizen_ats['ReturnDate']
        loan_period = abs(lend_diff.mean()).days
        return loan_period
    return 0

def get_number_ats(row, ats):
    citizen_ats = ats.loc[(ats['CitizenId'] == row['RealId']) & \
                (ats['LendDate'].dt.date <= row['RealDate'])]
    return len(citizen_ats)

def get_number_fall(row, falls):
    citizen_falls = falls.loc[(falls['CitizenId'] == row['RealId']) & \
            (falls['FallDate'].dt.date <= row['RealDate'])]
    return len(citizen_falls)

def get_hc_features(df: pd.DataFrame, citizen_idx_map: dict,
                     dates: np.ndarray, date_dict: dict, care_dict: dict) -> np.ndarray:
    """
    Encodes all observations for a single person to a sequence of variables containing
    the number of minutes of care received of that type.
    :param df: All observations for a single person (ID)
    :param citizen_idx_map: a mapping from citizen id to index
    :param care_dict: the care dictionary
    :return: An array where the two first columns are the year and week
    and the rest correspond to each type of care elements are number of minutes
    of care received in that week of that type of care.
    """
    minutes_care = np.zeros((len(citizen_idx_map), len(dates), len(care_dict)), np.float32)
    home_care = list(df[['CitizenId', 'Year', 'Week', 'CareType', 'Minutes', 'NumCares']].to_records(index=False))
    for obs in home_care:
        minutes_care[citizen_idx_map[obs[0]], date_dict[(obs[1], obs[2])],
                     care_dict[obs[3]]] += obs[4]*obs[5]
    return np.around(minutes_care)