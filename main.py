import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pth
import seaborn as sns

class CorrosionData:
        def __init__(self, path: 'str' = None, data: pd.DataFrame = None) -> None:
                self._data = data
                if data is None:
                        self.path = pth.Path(path)
                elif data is None and path is None:
                        raise ValueError('Both data and path are empty!')


                self._data_filtered = None

                self._column_names  = ['Unix Time (s)', 'Test Time (h)', 'Air Temp (C)', 'RH (%)',
                                      'Surface Temp (C)' ,'Cond Lo Freq (S)','Cond Hi Freq (S)',
                                      'Galv Corr 1 (A)','Galv Corr 2 (A)','Tot Cond Lo Freq (C/V)',
                                      'Tot Cond Hi Freq (C/V)','Tot Galv Corr 1 (C)','Tot Galv Corr 2 (C)']
                self.coeffs = {
                        'free_corrosion': 17.7059,
                        'galvanic_corrosion': 21.661
                }

        def loadData(self) -> pd.DataFrame:
                if self._data is None:
                        if not self.path.exists():
                                raise FileNotFoundError(f'File {self.path} does not exist')
                        self._data = pd.read_csv(self.path, header = None)
                        self._data = self._data.iloc[1:, :]

                self._data.columns = self._column_names
                self._convertTime()
                self._convert2numeric()

                return self._data

        @property
        def column_names(self):
                self._column_names = self._data.columns
                return self._column_names

        @property
        def data(self):
                return self._data

        def _convert2numeric(self) -> pd.DataFrame:
                self._data.iloc[:, 1:] = self._data.iloc[:, 1:].apply(pd.to_numeric, errors='raise')
                return self._data

        def _convertTime(self) -> pd.DataFrame:
                if 'Unix Time (s)' in self._data.columns:
                        self._data['Unix Time (s)'] = pd.to_datetime(self._data['Unix Time (s)'], unit='s')
                        self._data = self._data.rename(columns={'Unix Time (s)': 'Date-Time'})

                return self._data

        def limit_byDateTime(self, date: list[str]) -> pd.DataFrame:
                target_date = pd.to_datetime(date)
                self._data['Date-Time'] = pd.to_datetime(self._data['Date-Time'])
                data_filtered = self._data[self._data['Date-Time'].dt.date.isin(target_date.dt.date)]

                return data_filtered

        def select_byColumnNames(self, column_names: list[str]) -> pd.DataFrame:
                data_filtered = self._data[column_names]

                return data_filtered

        def add_NewColumn(self, column2apply: str, new_column_name: str, func):

                self._data[new_column_name] = self._data[column2apply].apply(func)

        def Compute_DailyAverages(self, time_name = 'Date-Time'):

                non_time_cols = self._data.iloc[:, 2:]
                non_time_cols = non_time_cols.columns
                data_avg = pd.DataFrame(self._data.groupby(self._data[time_name].dt.date)[non_time_cols].mean())
                data_time = pd.Series(self._data[time_name].groupby(self.data[time_name].dt.date), name=time_name)

                data_avg = pd.concat([data_time, data_avg], axis=1)

                return data_avg

        def Compute_1dAverages(self, time_name = 'Date-Time'):

                # Konwersja kolumny Date-Time do datetime, jeśli jeszcze nie jest w tym formacie
                self._data['Date-Time'] = pd.to_datetime(self._data['Date-Time'])

                # Szukamy pierwszego rekordu między 07:00 a 08:00
                mask_first = (self._data['Date-Time'].dt.hour >= 7) & (self._data['Date-Time'].dt.hour < 8)
                first_valid_time = self._data.loc[mask_first, 'Date-Time'].min() if mask_first.any() else self._data[
                        'Date-Time'].min()

                # Szukamy ostatniego rekordu między 06:00 a 07:00
                mask_last = (self._data['Date-Time'].dt.hour >= 6) & (self._data['Date-Time'].dt.hour < 7)
                last_valid_time = self._data.loc[mask_last, 'Date-Time'].max() if mask_last.any() else self._data[
                        'Date-Time'].max()
                data_avg = self._data.copy()
                # Przycinanie danych do zakresu [first_valid_time, last_valid_time]
                data_avg = data_avg[
                        (data_avg['Date-Time'] >= first_valid_time) & (data_avg['Date-Time'] <= last_valid_time)]

                data_avg['Date-Time'] = self._data['Date-Time'].dt.strftime('%H:%M')
                data_avg['Date-Time'] = data_avg['Date-Time'].dt.hour
                data_avg = data_avg.groupby(data_avg['Date-Time'])





        def plot_parameters(self, data, x_param_name: str, y_params: list[str] = None) -> None:
                """Plot selected parameters as a function of the chosen parameter."""


                if y_params is None:
                        # If no specific parameters are provided, plot all columns except the chosen one
                        y_params = [col for col in self.column_names if col != x_param_name]

                column_x, x_label = self._getColumn(x_param_name)


                plt.figure(figsize=(10, 6))

                # Plot each of the parameters against the chosen parameter
                for y_param in y_params:
                        column_y, y_label = self._getColumn(y_param)
                        plt.plot(column_x, column_y, label = y_label)


                plt.xlabel(x_label)
                plt.legend()
                plt.tight_layout()
                plt.grid(True)
                plt.show()

class Analyzer:
        def __init__(self, data: pd.DataFrame) -> None:
                self.data = data

        def find_correlations(self):
                correlation_matrix = self.data.corr()  # Compute correlation matrix
                plt.figure(figsize=(10, 10))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
                plt.title("Correlation Matrix")
                plt.xticks(rotation=45, ha='right')  # 'ha' ensures alignment
                plt.yticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

                return correlation_matrix
        
def main():
        path = 'Acuity_LS_00833_20250226_102627.csv'
        data_obj = CorrosionData(path)
        data_obj.loadData()
        data_obj.add_NewColumn(column2apply='Galv Corr 1 (A)',
                               new_column_name='Galv Corr Mass Loss Rate (g/m-a)',
                               func=lambda x: x*data_obj.coeffs['galvanic_corrosion'])

        data_obj.add_NewColumn(column2apply='Galv Corr 2 (A)',
                               new_column_name='Free Corr Mass Loss Rate (g/m-a)',
                               func=lambda x: x*data_obj.coeffs['galvanic_corrosion'])

        data_obj.add_NewColumn(column2apply='Tot Galv Corr 1 (C)',
                               new_column_name='Tot Galv Corr Mass Loss Rate (g/m-a)',
                               func=lambda x: x*data_obj.coeffs['galvanic_corrosion'])

        data_obj.add_NewColumn(column2apply='Tot Galv Corr 2 (C)',
                               new_column_name='Tot Free Corr Mass Loss Rate (g/m-a)2',
                               func=lambda x: x*data_obj.coeffs['free_corrosion'])

        print('--- COLUMN NAMES ---\n')
        for i, name in enumerate(data_obj.column_names):
                print(f'index: {i}, name: {name}')
        print(data_obj.column_names)

        data_obj.Compute_1dAverages('Date-Time')


        data_avg = data_obj.Compute_DailyAverages()
        data_obj_avg = CorrosionData(data=data_avg)

        data_selected = data_obj_avg.select_byColumnNames(['Air Temp (C)', 'RH (%)', 'Surface Temp (C)',
                                                           'Cond Lo Freq (S)', 'Cond Hi Freq (S)',
                                                           'Galv Corr 1 (A)', 'Galv Corr 2 (A)',
                                                           'Tot Cond Lo Freq (C/V)', 'Tot Cond Hi Freq (C/V)',
                                                           'Tot Galv Corr 1 (C)', 'Tot Galv Corr 2 (C)',
                                                           'Galv Corr Mass Loss Rate (g/m-a)', 'Free Corr Mass Loss Rate (g/m-a)',
                                                           'Tot Galv Corr Mass Loss Rate (g/m-a)', 'Tot Free Corr Mass Loss Rate (g/m-a)2'])

        analyzer_obj = Analyzer(data_selected)
        cor = analyzer_obj.find_correlations()



if __name__ == '__main__':
        main()