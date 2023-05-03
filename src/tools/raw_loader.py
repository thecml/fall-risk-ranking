from pathlib import Path
import pandas as pd
import numpy as np

class RawLoader:
    """
    Class for loading the raw data

    :param path: Path to the folder where the raw data is located
    """

    def __init__(self):
        pass

    def load_iso_classes(self, filename, path) -> pd.DataFrame:
        return pd.read_csv(Path.joinpath(path, filename), usecols=[0, 1, 2],
                           names=['DevISOClass', 'GroupSize', 'Description'],
                           encoding='iso-8859-10', converters={'DevISOClass': str})

    def load_assistive_aids(self, filename, path) -> pd.DataFrame:
        """
        This method loads assistive aids data
        :param filename: The name of the file with the data
        :return: A panda dataframe
        """
        converters = {'Personnummer': str, 'Kategori ISO nummer': str}
        df = pd.read_csv(Path.joinpath(path, filename), converters=converters,
                         encoding='iso-8859-10', skiprows=2)
        df = df.replace(r'^\s*$', np.nan, regex=True) # replace empty strs with nan
        df = df.dropna(subset=['Personnummer'])

        # Convert PN to CitizenId
        df['CitizenId'] = df['Personnummer'].str.replace("-", "") \
                            .astype(np.int64) \
                            .apply(lambda x: ((x*8) + 286) * 3) \
                            .astype(str)
        df = df.reset_index(drop=True)

        # Do some renaming
        df = df.rename(columns={'Kategori ISO nummer': 'DevISOClass',
                                'Leveret dato': 'LendDate',
                                'Returneret dato': 'ReturnDate'})
        df = df[['CitizenId', 'DevISOClass', 'LendDate', 'ReturnDate']]

        df['LendDate'] = pd.to_datetime(df['LendDate'], format='%d-%m-%Y')
        df['ReturnDate'] = pd.to_datetime(df['ReturnDate'], format='%d-%m-%Y', errors='coerce')

        return df

    def load_home_care(self, filenames, path) -> pd.DataFrame:
        """
        Parser for the DigiRehab home-care data that holds the number of minutes of home care received for each type
        of care. The parameters are:
         - Year
         - week
         - care type
         - Organisation (Private / municipal)
         - Minutes - Minutes of home care given that week
         - NumCares - how many visits of the given type carried out in the given week
         - sex
         - ID
         - BirthYear

        :param filename: The name of the file with the data
        :return: A panda dataframe
        """
        total_hc = pd.DataFrame()
        for filename in filenames:
                converters = {'Personnummer': str}
                if filename == 'HC2.csv' or filename == 'HC3.csv':
                    encoding = 'latin-1'
                else:
                    encoding = None
                df = pd.read_csv(Path.joinpath(path, filename),
                                 encoding=encoding,
                                 converters=converters,
                                 skiprows=2)
                df = df[df['Personnummer'].str.len() == 11] # remove empty/whitespace strings

                # Convert PN to CitizenId
                df['CitizenId'] = df['Personnummer'].str.replace("-", "").astype(np.int64) \
                                  .apply(lambda x: ((x*8) + 286) * 3).astype(str)

                # Calculate gender and birth year
                df['Gender'] = df['Personnummer'].str.replace("-", "").astype(np.int64) \
                               .apply(lambda x: 'FEMALE' if x % 2 == 0 else 'MALE')
                df['BirthYear'] = df['Personnummer'].str.replace("-", "") \
                                  .str.slice(4,6).astype(int)

                # Do some renaming
                df = df.rename(columns={'År uge': 'Year', 'Ugenummer': 'Week',
                                        'Ydelse navn' : 'CareType',
                                        'Leveret tid (minutter)': 'Minutes',
                                        'Antal ydelser': 'NumCares'})

                # Fix year, convert types
                df['Year'] = [int(x.split('-')[0]) for x in df.Year]
                df = df[['CitizenId', 'Gender', 'BirthYear', 'Year', 'Week',
                         'Minutes', 'NumCares', 'CareType']]

                total_hc = pd.concat([total_hc, df], ignore_index=True)
        return total_hc