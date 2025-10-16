import pathlib as pth
import argparse
from typing import Union, Optional
import pandas as pd

from AnalyzingTools import CorrosionData, VisualizeData
from collections import OrderedDict


def loadRawData(path: Union[str, pth.Path]) -> CorrosionData:
        path = pth.Path(path)
        
        
        column_names = ['Unix Time (s)', 'Test Time (h)', 'Air Temp (C)', 'RH (%)',
                              'Surface Temp (C)', 'Cond Lo Freq (S)', 'Cond Hi Freq (S)',
                              'Galv Corr 1 (A)', 'Galv Corr 2 (A)', 'Tot Cond Lo Freq (C/V)',
                              'Tot Cond Hi Freq (C/V)', 'Tot Galv Corr 1 (C)', 'Tot Galv Corr 2 (C)']

        data_obj = CorrosionData(path=path, column_names=column_names, data=None)
        print(f'\nLoaded data from path {path}\n')

        return data_obj



def plot():
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


def argparser():
        file_name = 'Acuity_LS_00833_20250226_102627.csv'
        base_path = pth.Path(__file__).parent.parent
        file_path = base_path / 'data' / file_name

        print(file_path)
        
        parser = argparse.ArgumentParser(
        description="Script for training the model based on predefined range of scenarios",
        formatter_class=argparse.RawTextHelpFormatter
        )

        parser.add_argument(
                '-p', '--path',
                type=str,
                default=str(file_path.resolve()),
                help=(
                'Absolute path to the data file\n'
                f'If None it defaults to first file in {file_path}')
                )
        
            # Flag definition
        parser.add_argument(
                '-d', '--device',
                type=str,
                default='cpu',
                choices=['cpu', 'cuda', 'gpu'], # choice limit
                help=(
                "Device for tensor based computation.\n"
                "Pick 'cpu' or 'cuda'/ 'gpu'.\n"
                )
        )   


        args = parser.parse_args()
        return args

def _get_plot_params() -> dict:
        params = OrderedDict()
        iter = 0
        

        params = OrderedDict()
        while True:
                print(f'ITERATION: {iter}')

                x = input(f'Input x parameter name (leave empty to finish): ').strip()
                y = input(f'Input y parameter names (comma-separated): ').strip().split(',')
                params[x] = y

                stop = False if input('Stop? (y/n): ').strip().lower() == 'n' else True
                
                if stop:
                        break
        return params
        
        

def main():
        args = argparser()
        
        

        print(67*'=')
        print('Welcome to Corrosion Data Analysis Tool')
        print("This program helps analyze and visualize corrosion measurement data")
        print(67*'=')

        data_obj = loadRawData(args.path)


        available_options = {
                0: 'Load new data from path',
                1: 'Add new column',
                2: 'Compute average from every day',
                3: 'Compute single, average day from data',
                4: 'Select specific columns',
                5: 'Reset data to one thats been loaded from file',
                6: 'Plot parameters',
                7: 'Print data table',
                8: 'Exit'
        }


        
        data_obj0 = None
        visualiser = None

        while True:

                data_obj.add_NewColumn(column2apply='Galv Corr 1 (A)', # add new column based on formula - Galv Corr 1 (A) is x, only one variable for now
                        new_column_name='Galv Corr Mass Loss Rate (g/m-a)',
                        func=lambda x: x * data_obj.coeffs['galvanic_corrosion'])
        
                data_obj.add_NewColumn(column2apply='Tot Galv Corr 1 (C)',
                                new_column_name='Tot Galv Corr Mass Loss Rate (g/m-a)',
                                func=lambda x: x * data_obj.coeffs['galvanic_corrosion'])

                print('\n--- COLUMN NAMES ---\n') # wypisanie nazw kolumn
                for i, name in enumerate(data_obj.column_names):
                        print(f'index: {i}, name: {name}')

                # more comfy to copy from here
                print('\n--- COLUMN NAMES (TO COPY PART OF LIST) ---')
                print(data_obj.column_names) # wypisanie nazw kolumn

                print(20*'=')

                print("Available options:")
                for key, value in available_options.items():
                        print(f"{key}: {value}")

                try:
                        choice = int(input("\nPick option: ").strip())
                except Exception as _:
                        print('Invalid choice (not an integer). Please try again.')
                        continue

                match choice:
                        case 0:
                                path = input('Input new absolute path to data file: ')
                                data_obj = loadRawData(path)
                                data_obj0 = data_obj.deepcopy()

                        case 1:
                                column2apply = input('Input column name (str) to apply formula to: ')
                                new_column_name = input('Input new column name (str): ')
                                func = input('Input formula (eg. lambda x: edit this -> function(x), where x is column to apply)')

                                data_obj.add_NewColumn(column2apply, new_column_name, func)
                        case 2:
                                data_obj.Compute_DailyAverages()
                        case 3:
                                data_obj.Compute_OneDayAverage()
                        case 4:
                                column_names = input('Input column names to select (list[str]): ')
                                data_obj.select_byColumnNames(column_names)
                        case 5:
                                data_obj = data_obj0.deepcopy()
                        case 6:
                                visualiser = VisualizeData(data_obj.data, data_obj.path.stem)
                                params2plot = _get_plot_params()

                                visualiser.plot_parameters(params2plot)
                                
                                
                        case 7:
                                visualiser = VisualizeData(data_obj.data, data_obj.path.stem)
                                visualiser.view_dataframe()
                                


                        case 8:
                                print('Exiting...')
                                break
                        case _:
                                print('Invalid choice. Please try again.')
                







if __name__ == '__main__':
        main() # główna część kodu patrz tutaj