import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pth
import seaborn as sns

class CorrosionData:
        def __init__(self, path: 'str' = None, column_names = None, data: pd.DataFrame = None) -> None:
                self._data_filtered = None

                self._column_names  = column_names

                self.coeffs = {
                        'free_corrosion': 17.7059,
                        'galvanic_corrosion': 21.661
                }


                self._data = data
                if data is None and path is not None:
                        self.path = pth.Path(path)
                        self._data = self.loadData()
                elif data is not None:
                        self._data.dropna()
                elif data is None and path is None:
                        raise ValueError('Both data and path are empty!')




        def loadData(self) -> pd.DataFrame:
                if self._data is None:
                        if not self.path.exists():
                                raise FileNotFoundError(f'File {self.path} does not exist')
                        self._data = pd.read_csv(self.path, header = None)
                        self._data = self._data.iloc[1:, :]

                self._data.columns = self._column_names
                self._convertTime()
                self._convert2numeric()

                self._data.dropna()
                return self._data

        @property
        def column_names(self):
                self._column_names = self._data.columns
                return self._column_names

        @property
        def data(self):
                return self._data
        @property
        def data_numeric(self):
                return self._data._get_numeric_data()

        def _convert2numeric(self) -> pd.DataFrame:
                self._data.iloc[:, 1:] = self._data.iloc[:, 1:].apply(pd.to_numeric, errors='raise')
                return self._data

        def _convertTime(self) -> pd.DataFrame:
                if 'Unix Time (s)' in self._data.columns:
                        self._data['Unix Time (s)'] = pd.to_numeric(self._data['Unix Time (s)'])
                        self._data['Unix Time (s)'] = pd.to_datetime(self._data['Unix Time (s)'], unit='s')
                        self._data = self._data.rename(columns={'Unix Time (s)': 'Date-Time'})

                return self._data


        # TE FUNKCJE SĄ DO UŻYCIA W NOWYM KODZIE
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
                non_time_col_names= non_time_cols.columns

                data_avg = non_time_cols.groupby(self._data['Date-Time'].dt.date).mean()


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

                # data_avg['Date-Time'] = self._data['Date-Time'].dt.strftime('%H:%M')

                # Dodajemy kolumnę godzinową
                data_avg['Date-Time'] = data_avg[time_name].dt.hour # time albo hour


                # Grupowanie po godzinach
                data_avg = data_avg.groupby('Date-Time').mean()


                data_avg = data_avg.rename(columns={'Date-Time': 'Hour'})

                data_avg = data_avg.drop(columns = ['Test Time (h)'])

                return data_avg

        def plot_Trend(self, x: np.array, y: np.array, poly_order: int = 4) -> object:
                # Fit a linear trend line (1st-degree polynomial)
                coeffs = np.polyfit(x, y, poly_order)  # Linear fit
                trend_line = np.poly1d(coeffs)

                return trend_line

        def sort_cols(self, x, y) -> tuple[np.array, np.array]:
                x = np.asarray(x)
                y = np.asarray(y)

                sorted_indices = np.argsort(x)  # Get sorted indices of the first row
                x = x[sorted_indices]
                y = y[sorted_indices]

                x = x.tolist()
                y = y.tolist()

                return x, y

        def make_lists(self, x, y):
                x = np.asarray(x)
                y = np.asarray(y)

                x = x.tolist()
                y = y.tolist()

                return x, y


        def plot_parameters(self, x_param_name: str, y_params: list[str] = None, trend_line: bool = True, sort_x: bool = True) -> None:
                """Plot selected parameters as a function of the chosen parameter."""

                if y_params is None:
                        # If no specific parameters are provided, plot all columns except the chosen one
                        y_params = [col for col in self.column_names if col != x_param_name]

                column_x = self._data[x_param_name]
                x_label = x_param_name


                plt.figure(figsize=(10, 6))

                # Plot each of the parameters against the chosen parameter
                for y_param in y_params:
                        column_y = self._data[y_param]
                        y_label = y_param
                        if sort_x is True:
                                column_x, column_y = self.sort_cols(column_x, column_y)
                        else:
                                column_x, column_y = self.make_lists(column_x, column_y)

                        plt.scatter(column_x, column_y, label = y_label, marker = 'o')
                        if trend_line is True:
                                try:
                                        trend_eq = self.plot_Trend(column_x, column_y)

                                        plt.plot(column_x, trend_eq(column_x), linestyle= '-.', label = f'Trend Line of {y_label}')
                                except Exception as e:
                                        print(f'Error when computing trend line: {e}.\nThis behavior is normal for irregular data')


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

def loadRawData(path):
        column_names = ['Unix Time (s)', 'Test Time (h)', 'Air Temp (C)', 'RH (%)',
                              'Surface Temp (C)', 'Cond Lo Freq (S)', 'Cond Hi Freq (S)',
                              'Galv Corr 1 (A)', 'Galv Corr 2 (A)', 'Tot Cond Lo Freq (C/V)',
                              'Tot Cond Hi Freq (C/V)', 'Tot Galv Corr 1 (C)', 'Tot Galv Corr 2 (C)']


        data_obj = CorrosionData(path=path, column_names=column_names)
        return data_obj



def main():
        path = 'Acuity_LS_00833_20250226_102627.csv'
        """
        Wczytanie danych jako DataFrame i stworzenie z nich obiektu data_obj. 
        data_obj jest jak data_frame (zbiór danych), ale z kilkoma dopisanymi metodami (funckjami). 
        Metody, których warto używać podczas analiz mają postać data_obj.funkcja(argumenty).
        Metody o postaci data_obj._funkcja(argumenty) odpowiadają za wewnętrzne działanie kodu. Nie trzeba ich ruszać.
        
        Da się "dostać" do danych zawartych w data_obj:
        data = data_obj.data <- jako data zadeklarujemy DataFrame z funkcjonalnością taką jak w bibliotece pandas
        
        alternatywnie te same efekty można uzyskać poprzez:
        data_obj.data.wybrana_metoda_z_pandas
        """
        data_obj = loadRawData(path) #
        """
        data_obj.add_NewColumn() dodaje nową kolumnę do datasetu. Dodanie kolumn czasowych będzie wymagało modyfikacji kodu!!!!
        argumenty:
        column2apply: Kolumna będąca argumentem do wzoru w func
        new_column_name: Nazwa nowej kolumny
        func: funkcja na podstawie której powstaje nowa kolumna
                Funkcja ma konstrukcję lambda (oznaczenie funkcji) x (argument na której działa funkcja): wzór funkcji(x)
        """
        data_obj.add_NewColumn(column2apply='Galv Corr 1 (A)',
                               new_column_name='Galv Corr Mass Loss Rate (g/m-a)',
                               func=lambda x: x * data_obj.coeffs['galvanic_corrosion'])

        data_obj.add_NewColumn(column2apply='Tot Galv Corr 1 (C)',
                               new_column_name='Tot Galv Corr Mass Loss Rate (g/m-a)',
                               func=lambda x: x * data_obj.coeffs['galvanic_corrosion'])


        print('--- COLUMN NAMES ---\n') # wypisanie nazw kolumn
        for i, name in enumerate(data_obj.column_names):
                print(f'index: {i}, name: {name}')

        """
        metoda data_obj.Compute_1dAverages() zwraca nowy data frame, 
        który zawiera dane uśrednione: powstaje jeden dzień średni dla dni zawartych we wszystkich rekordach
        """
        data_avg = data_obj.Compute_DailyAverages()
        """
        metoda data_obj.Compute_1dAverages() zwraca nowy data frame, 
        który zawiera dane uśrednione: każdy dzień jest sprowadzony do jednego punktu pomiarowego - średniej z tego dnia
        """
        data_avg = data_obj.Compute_DailyAverages() # each day averaged to a single data point


        """
        powyższe metody zwracają nowy DataFrame (nową "tabelkę z danymi")
        Żeby dalej je analizować z wykorzystaniem CorrosionData trzeba stworzyć nowy obiekt
        """
        data_obj_avg = CorrosionData(data=data_avg)
        """
        Da się stworzyć DataFrame tylko z wybranymi kolumnami żeby pracowało się z nimi wygodniej:
        data_obj_avg.select_byColumnNames zwróci DataFrame z wybranymi nazwami kolumn.
        Metoda przyjmuje listę nazw kolumn: list[str]
        """
        # data_selected = data_obj_avg.select_byColumnNames(['Air Temp (C)', 'RH (%)', 'Surface Temp (C)',
        #                                                    'Cond Lo Freq (S)', 'Cond Hi Freq (S)',
        #                                                    'Galv Corr 1 (A)', 'Galv Corr 2 (A)',
        #                                                    'Tot Cond Lo Freq (C/V)', 'Tot Cond Hi Freq (C/V)',
        #                                                    'Tot Galv Corr 1 (C)', 'Tot Galv Corr 2 (C)',
        #                                                    'Galv Corr Mass Loss Rate (g/m-a)', 'Free Corr Mass Loss Rate (g/m-a)',
        #                                                    'Tot Galv Corr Mass Loss Rate (g/m-a)', 'Tot Free Corr Mass Loss Rate (g/m-a)2'])


        """
        data_obj_avg.plot_parameters() służy do tworzenia wykresów. Wybieramy argument osi x i listę kolumn na osi y 
        (jeszcze nie da się ustawić kilku osi z oddzielnym skalowaniem :c)
        trend_line = True: dodatkowo wyrysuje linię trendu (aproksymację wielomianową)
        sort_x = False: czasami rekordy na osi x mogą zmieniać się w sposób niemonotoniczny i wykres może nie wyjść. 
        sort_x sortuje wartości kolumn i umożliwia sprawdzenie trendu w zależności argumentów.
        """
        data_obj_avg.plot_parameters(x_param_name = 'Air Temp (C)', y_params=['Surface Temp (C)'],
                                     trend_line=True, sort_x=True)

        """
        Analyzer obiekt do alternatywnych analiz danych. Do inicjalizacji jest potrzebny DataFrame
        Póki co jedyną metodą jest .find_correlations
        """
        analyzer_obj = Analyzer(data_obj_avg.select_byColumnNames(['Air Temp (C)', 'RH (%)', 'Surface Temp (C)',
                                                                   'Tot Cond Lo Freq (C/V)', 'Tot Cond Hi Freq (C/V)',
                                                                   'Tot Galv Corr 1 (C)', 'Tot Galv Corr 2 (C)',
                                                                   'Galv Corr Mass Loss Rate (g/m-a)',
                                                                   'Tot Galv Corr Mass Loss Rate (g/m-a)']))

        """
        find_correlations umożliwia wyrysowanie macierzy korelacji dla wybranych (zawartych w dataframe kolumn)
        dodatkowo metoda zwraca tekstową formę macierzy
        """
        cor = analyzer_obj.find_correlations()



if __name__ == '__main__':
        main() # główna część kodu patrz tutaj