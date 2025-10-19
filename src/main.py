import pathlib as pth
from typing import Union, Optional
import pandas as pd

import ast

from AnalyzingTools import CorrosionData, VisualizeData
from FileExplorer import pick_file



def loadRawData(path: Union[str, pth.Path]) -> CorrosionData:
        path = pth.Path(path)
        
        
        column_names = ['Unix Time (s)', 'Test Time (h)', 'Air Temp (C)', 'RH (%)',
                              'Surface Temp (C)', 'Cond Lo Freq (S)', 'Cond Hi Freq (S)',
                              'Galv Corr 1 (A)', 'Galv Corr 2 (A)', 'Tot Cond Lo Freq (C/V)',
                              'Tot Cond Hi Freq (C/V)', 'Tot Galv Corr 1 (C)', 'Tot Galv Corr 2 (C)']

        data_obj = CorrosionData(path=path, column_names=column_names, data=None)
        print(f'\nLoaded data from path {path}\n')

        return data_obj

def _get_plot_params() -> dict:
    """
    Get plot parameters from user input.
    Returns dict with keys: 'x_axis', 'left_axis', 'right_axis' (optional), and 'sort_x'
    """
    x_axis = input('Input x parameter name (str): ').strip()
    
    left_axis = input('Input left axis parameter names (list[str]): ')
    left_axis = ast.literal_eval(left_axis)
    
    sort_x = input('Sort by x values? (y/n): ').strip().lower() == 'y'
    
    params = {
        'x_axis': x_axis,
        'left_axis': left_axis,
        'sort_x': sort_x
    }
    
    use_right = input('Use right axis? (y/n): ').strip().lower() == 'y'
    if use_right:
        right_axis = input('Input right axis parameter names (list[str]): ')
        right_axis = ast.literal_eval(right_axis)
        params['right_axis'] = right_axis
    
    return params

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        path: pth.Path = pick_file()
        data_obj = loadRawData(path)
        data_obj0 = data_obj.deepcopy()

        return data_obj, data_obj0
               

def data_loop():
        

        print(67*'=')
        print('Welcome to Corrosion Data Analysis Tool')
        print("This program helps analyze and visualize corrosion measurement data")
        print(67*'=')
        
        data_obj, data_obj0 = load_data()

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
                                data_obj, data_obj0 = load_data()

                                print(data_obj)

                        case 1:
                                column2apply = input('Input column name (str) to apply formula to: ')
                                new_column_name = input('Input new column name (str): ')
                                func = input('Input formula (eg. lambda x: edit this -> function(x), where x is column to apply)')

                                data_obj.add_NewColumn(column2apply, new_column_name, func)
                        case 2:
                                data_obj = data_obj0.deepcopy()
                                data_obj.Compute_DailyAverages()
                        case 3:
                                data_obj = data_obj0.deepcopy()
                                data_obj.Compute_OneDayAverage()
                        case 4:
                                column_names = ast.literal_eval(input('Input column names to select (list[str]): '))
                                data_obj.select_byColumnNames(column_names)
                        case 5:
                                data_obj = data_obj0.deepcopy()
                        case 6:
                                visualiser = VisualizeData(data_obj._data, data_obj.path.stem)
                                params2plot = _get_plot_params()
                                sort_x = params2plot.pop('sort_x', False)

                                visualiser.plot_parameters(params2plot, sort_x=sort_x)
                                
                                
                        case 7:
                                visualiser = VisualizeData(data_obj.data, data_obj.path.stem)
                                visualiser.view_dataframe()
                                


                        case 8:
                                print('Exiting...')
                                break
                        case _:
                                print('Invalid choice. Please try again.')
                

def main():
        data_loop()

if __name__ == '__main__':
        main() # główna część kodu patrz tutaj