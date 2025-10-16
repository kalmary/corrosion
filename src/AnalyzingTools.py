from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import pathlib as pth
import seaborn as sns

import tkinter as tk
import pandas as pd
from pandastable import Table, TableModel


import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from PyQt5.QtWidgets import QApplication




class CorrosionData:
    def __init__(self, path: Optional[Union[str, pth.Path]], column_names: Optional[list[str]], data: Optional[pd.DataFrame]) -> None:
        self._data_filtered = None

        self._column_names = column_names

        self.coeffs = {
            'free_corrosion': 17.7059,
            'galvanic_corrosion': 21.661
        }

        self._data = data
        if data is not None:
            self._data = data
            self._data.dropna()
        elif data is None and path is not None:
            self.path = pth.Path(path)
            self._data = self.loadData(self.path)
        elif data is None and path is None:
            raise ValueError('Both data and path are empty!')

    def loadData(self, path) -> pd.DataFrame:
        if self._data is None:
            if not path.exists():
                raise FileNotFoundError(f'File {path} does not exist')
            self._data = pd.read_csv(path, header=None)
            self._data = self._data.iloc[1:, :]

        if self._column_names is not None:
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


    def limit_byDateTime(self, date: list[str]) -> pd.DataFrame:
        target_date = pd.to_datetime(date)
        self._data['Date-Time'] = pd.to_datetime(self._data['Date-Time'])
        data_filtered = self._data[self._data['Date-Time'].dt.date.isin(target_date.dt.date)]

        return data_filtered

    def select_byColumnNames(self, column_names: list[str]) -> pd.DataFrame:
        data_filtered = self._data[column_names]

        self._data = data_filtered

        return data_filtered

    def add_NewColumn(self, column2apply: str, new_column_name: str, func):

        self._data[new_column_name] = self._data[column2apply].apply(func)

    def Compute_DailyAverages(self, time_name='Date-Time'):

        non_time_cols = self._data.iloc[:, 2:]
        non_time_col_names = non_time_cols.columns

        data_avg = non_time_cols.groupby(self._data['Date-Time'].dt.date).mean()

        self._data = data_avg

        return data_avg

    def Compute_OneDayAverage(self, time_name: str ='Date-Time'):

        # conversion to date-time if its not already done
        if 'Date-Time' not in self._data.columns:
            self._convertTime()
        self._data['Date-Time'] = pd.to_datetime(self._data['Date-Time'])

        # look for first record from 7 am - 8 am
        mask_first = (self._data['Date-Time'].dt.hour >= 7) & (self._data['Date-Time'].dt.hour < 8)
        first_valid_time = self._data.loc[mask_first, 'Date-Time'].min() if mask_first.any() else self._data[
            'Date-Time'].min()

        # look for last record from 6 am - 7 am
        mask_last = (self._data['Date-Time'].dt.hour >= 6) & (self._data['Date-Time'].dt.hour < 7)
        last_valid_time = self._data.loc[mask_last, 'Date-Time'].max() if mask_last.any() else self._data[
            'Date-Time'].max()
        
        data_avg = self._data.copy()

        # cut data to fit desired range
        data_avg = data_avg[
            (data_avg['Date-Time'] >= first_valid_time) & (data_avg['Date-Time'] <= last_valid_time)]

        # data_avg['Date-Time'] = self._data['Date-Time'].dt.strftime('%H:%M')

        # add only 
        data_avg['Date-Time'] = data_avg[time_name].dt.hour  # time albo hour


        data_avg = data_avg.groupby('Date-Time').mean()

        data_avg = data_avg.rename(columns={'Date-Time': 'Hour'})

        data_avg = data_avg.drop(columns=['Test Time (h)'])

        self._data = data_avg

        return data_avg


class VisualizeData:
    def __init__(self, data: pd.DataFrame, data_name: Optional[str]) -> None:
        self.data = data
        self.data_name = data_name

    def view_dataframe(self):
        #main window
        root = tk.Tk()
        if self.data_name is not None:
            root.title(f"Data loaded from: {self.data_name}")

        #window size
        root.geometry("1000x600") 


        frame = tk.Frame(root)
        frame.pack(fill='both', expand=True)

        pt = Table(frame, 
                   dataframe=self.data,
                   showtoolbar=True,  # toolbar on top
                   showstatusbar=True) # status barr bottom

        # enable resize and scroll
        pt.autoResizeColumns()
        

        # display
        pt.show() 
        root.mainloop()

    def plot_parameters(self, params_dict: dict = None, trend_line: bool = True, 
                        sort_x: bool = True) -> None:
            """
            Plot parameters with interactive controls using PyQtGraph.
            
            Args:
                params_dict: Dictionary where keys are x-parameter names and values are lists of y-parameter names
                            Example: {'time': ['temperature', 'pressure'], 'distance': ['speed']}
                            If None, plots all numeric columns vs index
                            Special key None can be used for index-based x-axis
                trend_line: Whether to add polynomial trend lines
                sort_x: Whether to sort by x values
            """
            
            # Handle default case - plot all numeric columns vs index
            if params_dict is None:
                numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
                params_dict = {None: numeric_cols}
            
            # Create Qt Application if it doesn't exist
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)

            # Create window
            win = pg.GraphicsLayoutWidget(show=True)
            title = f"Interactive Plot"
            if self.data_name:
                title += f" - {self.data_name}"
            win.setWindowTitle(title)
            win.resize(1200, 700)

            # Count total y parameters to decide if we need dual axes
            total_y_params = sum(len(y_list) for y_list in params_dict.values())
            use_dual_axes = total_y_params > 1

            # Create main plot with left axis
            plot1 = win.addPlot()
            plot1.showGrid(x=True, y=True, alpha=0.3)
            
            # Set x-axis label
            if len(params_dict) == 1:
                x_key = list(params_dict.keys())[0]
                x_label = x_key if x_key is not None else "Index"
            else:
                x_label = "X Values (mixed parameters)"
            plot1.setLabel('bottom', x_label)
            plot1.setLabel('left', 'Left Axis' if use_dual_axes else 'Values')
            
            # Create second ViewBox for right axis if needed
            plot2 = None
            legend2 = None
            if use_dual_axes:
                plot2 = pg.ViewBox()
                plot1.showAxis('right')
                plot1.scene().addItem(plot2)
                plot1.getAxis('right').linkToView(plot2)
                plot2.setXLink(plot1)
                plot1.getAxis('right').setLabel('Right Axis')
                
                # Add legends
                legend1 = plot1.addLegend(offset=(10, 10))
                legend2 = pg.LegendItem(offset=(10, 100))
                legend2.setParentItem(plot1.graphicsItem())
            else:
                legend1 = plot1.addLegend(offset=(10, 10))
            
            # Define colors
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 
                    (255, 128, 0), (128, 0, 255), (0, 255, 128)]
            
            color_idx = 0
            param_count = 0
            
            # Plot each x-parameter with its y-parameters
            for x_param, y_params in params_dict.items():
                # Determine x values
                if x_param is None:
                    x_values = np.arange(len(self.data))
                else:
                    x_values = self._data[x_param].values
                
                for y_param in y_params:
                    y_values = self._data[y_param].values.copy()
                    x_vals = x_values.copy()
                    
                    # Alternate between left and right axis if using dual axes
                    use_right_axis = use_dual_axes and (param_count % 2 == 1)
                    
                    if sort_x and x_param is not None:
                        sort_idx = np.argsort(x_vals)
                        x_vals = x_vals[sort_idx]
                        y_values = y_values[sort_idx]
                    
                    color = colors[color_idx % len(colors)]
                    
                    # Choose which plot to add to
                    target_plot = plot2 if use_right_axis else plot1
                    target_legend = legend2 if use_right_axis else legend1
                    
                    # Create label for legend
                    if x_param is None:
                        label = y_param
                    else:
                        label = f'{y_param} vs {x_param}'
                    
                    # Add scatter plot
                    curve = pg.ScatterPlotItem(
                        x=x_vals, 
                        y=y_values,
                        pen=None,
                        symbol='o',
                        size=8,
                        brush=color
                    )
                    target_plot.addItem(curve)
                    target_legend.addItem(curve, label)
                    
                    # Add trend line if requested
                    if trend_line:
                        try:
                            mask = ~(np.isnan(x_vals) | np.isnan(y_values))
                            x_clean = x_vals[mask]
                            y_clean = y_values[mask]
                            
                            if len(x_clean) > 5:
                                coeffs = np.polyfit(x_clean, y_clean, min(4, len(x_clean)-1))
                                trend_eq = np.poly1d(coeffs)
                                
                                x_smooth = np.linspace(x_clean.min(), x_clean.max(), 200)
                                y_smooth = trend_eq(x_smooth)
                                
                                trend_curve = pg.PlotCurveItem(
                                    x=x_smooth,
                                    y=y_smooth,
                                    pen=pg.mkPen(color, style=pg.QtCore.Qt.DashLine, width=2)
                                )
                                target_plot.addItem(trend_curve)
                                target_legend.addItem(trend_curve, f'Trend: {y_param}')
                        except Exception as e:
                            print(f'Error computing trend line for {y_param}: {e}')
                    
                    color_idx += 1
                    param_count += 1
            
            # Update views when plot1 changes (if using dual axes)
            if use_dual_axes and plot2 is not None:
                def update_views():
                    plot2.setGeometry(plot1.vb.sceneBoundingRect())
                    plot2.linkedViewChanged(plot1.vb, plot2.XAxis)
                
                update_views()
                plot1.vb.sigResized.connect(update_views)
            
            # Enable auto-range
            plot1.enableAutoRange()
            
            # Start Qt event loop if needed
            if app is not None:
                app.exec_()


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
    
