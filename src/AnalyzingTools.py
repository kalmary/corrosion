from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import pathlib as pth
import seaborn as sns

import tkinter as tk
import pandas as pd
from pandastable import Table, TableModel
import copy

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from PyQt5.QtWidgets import QApplication
import sys




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
        # If no dataframe is loaded yet, return any previously set column names
        if self._data is None:
            return self._column_names

        # If data is a DataFrame, use its columns
        if isinstance(self._data, pd.DataFrame):
            self._column_names = self._data.columns
        # If data is a Series, convert to DataFrame first
        elif isinstance(self._data, pd.Series):
            self._column_names = self._data.to_frame().columns

        return self._column_names

    @property
    def data(self):
        return self._data

    @property
    def data_numeric(self) -> pd.DataFrame:
        """
        Return only numeric columns from the stored data, handling None and Series cases.
        Raises:
            ValueError: if no data is available.
        """
        if self._data is None:
            raise ValueError("No data available to return numeric columns")

        # If a Series is stored, convert to a DataFrame first
        if isinstance(self._data, pd.Series):
            df = self._data.to_frame()
            return df.select_dtypes(include=[np.number])

        # If already a DataFrame, select numeric columns
        if isinstance(self._data, pd.DataFrame):
            return self._data.select_dtypes(include=[np.number])

        # Fallback: attempt to coerce to a DataFrame and select numeric columns
        df = pd.DataFrame(self._data)
        return df.select_dtypes(include=[np.number])

    def _convert2numeric(self) -> pd.DataFrame:
        self._data.iloc[:, 1:] = self._data.iloc[:, 1:].apply(pd.to_numeric, errors='raise') # type: ignore
        return self._data # type: ignore

    def _convertTime(self) -> pd.DataFrame:
        # If there's no data, just return as-is to avoid "None is not subscriptable"
        if self._data is None:
            return self._data # type: ignore

        # Work with a DataFrame; if it's a Series, convert to DataFrame first
        if isinstance(self._data, pd.Series):
            df = self._data.to_frame()
        else:
            df = self._data

        # Proceed only if the expected column exists
        if hasattr(df, 'columns') and 'Unix Time (s)' in df.columns:
            # Coerce invalid values instead of raising, then convert to datetime
            df['Unix Time (s)'] = pd.to_numeric(df['Unix Time (s)'], errors='coerce')
            df['Unix Time (s)'] = pd.to_datetime(df['Unix Time (s)'], unit='s', errors='coerce')
            df = df.rename(columns={'Unix Time (s)': 'Date-Time'})

            # store back the possibly modified DataFrame
            self._data = df

        return self._data # type: ignore


    def limit_byDateTime(self, date: Union[str, list[str]]) -> pd.DataFrame:
        # Ensure data is loaded
        if self._data is None:
            raise ValueError("No data available to filter by date")

        # Ensure 'Date-Time' column exists (try to convert if not present)
        if 'Date-Time' not in getattr(self._data, 'columns', []):
            self._convertTime()
            if 'Date-Time' not in getattr(self._data, 'columns', []):
                raise KeyError("No 'Date-Time' column found in data")

        # Parse target dates; accept single string or list-like input
        target_dt = pd.to_datetime(date, errors='coerce')
        if isinstance(target_dt, (pd.DatetimeIndex, pd.Series)):
            target_dates = target_dt.date
        else:
            if pd.isna(target_dt):
                raise ValueError("Could not parse provided date(s)")
            target_dates = [target_dt.date()]

        # Ensure Date-Time column is datetime dtype
        self._data['Date-Time'] = pd.to_datetime(self._data['Date-Time'], errors='coerce')

        # Create boolean mask safely and return filtered copy
        mask = self._data['Date-Time'].dt.date.isin(target_dates)
        data_filtered = self._data.loc[mask].copy()

        return data_filtered

    def select_byColumnNames(self, column_names: list[str]) -> pd.DataFrame:
        data_filtered = self._data[column_names] # type: ignore

        self._data = data_filtered

        return data_filtered

    def add_NewColumn(self, column2apply: str, new_column_name: str, func):

        self._data[new_column_name] = self._data[column2apply].apply(func) # type: ignore

    def Compute_DailyAverages(self, time_name='Date-Time'):

        non_time_cols = self._data.iloc[:, 2:]
        non_time_col_names = non_time_cols.columns

        data_avg = non_time_cols.groupby(self._data['Date-Time'].dt.date).mean() # type: ignore

        self._data = data_avg

        return data_avg

    def Compute_OneDayAverage(self, time_name: str ='Date-Time'):

        # ensure data exists
        if self._data is None:
            raise ValueError("No data available to compute daily averages")

        # conversion to date-time if its not already done
        if 'Date-Time' not in self._data.columns:
            self._convertTime()

        # Safely convert Date-Time column to datetime, allowing coercion of invalid values
        self._data['Date-Time'] = pd.to_datetime(self._data['Date-Time'], errors='coerce')

        # Ensure there are valid datetime values
        if self._data['Date-Time'].isna().all():
            raise ValueError("No valid 'Date-Time' values available after conversion")

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

        # drop Test Time (h) only if present
        if 'Test Time (h)' in data_avg.columns:
            data_avg = data_avg.drop(columns=['Test Time (h)'])

        self._data = data_avg

        return data_avg

    def deepcopy(self):
        new_obj = CorrosionData(
            path=self.path if hasattr(self, 'path') else None,
            column_names=copy.deepcopy(self._column_names), # type: ignore
            data=self._data.copy()
        )
        if hasattr(self, 'path'):
            new_obj.path = self.path
        return new_obj


class VisualizeData:
    def __init__(self, data: pd.DataFrame, data_name: Optional[str]) -> None:
        self.data = data
        self.data_name = data_name

    def plot_parameters(self, params_dict: dict = None, sort_x: bool = False) -> None:
        """
        Plot parameters with interactive controls using PyQtGraph.
        
        Args:
            params_dict: Dictionary with keys 'x_axis', 'left_axis', and optionally 'right_axis'
                        Example: {'x_axis': 'time', 'left_axis': ['temp'], 'right_axis': ['pressure']}
                        If None, plots all numeric columns vs index
            sort_x: Whether to sort by x values
        """
        
        # Handle default case - plot all numeric columns vs index
        if params_dict is None:
            numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
            params_dict = {'x_axis': None, 'left_axis': numeric_cols}
        
        # Extract parameters
        x_param = params_dict.get('x_axis')
        left_params = params_dict.get('left_axis', [])
        right_params = params_dict.get('right_axis', [])
        
        # Combine all y parameters for processing
        all_y_params = left_params + right_params
        
        # Create Qt Application if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Create window with white background
        win = pg.GraphicsLayoutWidget(show=True)
        win.setBackground('w')
        title = f"Interactive Plot"
        if self.data_name:
            title += f" - {self.data_name}"
        win.setWindowTitle(title)
        win.resize(1200, 700)

        # Count total y parameters to decide if we need dual axes
        use_dual_axes = len(right_params) > 0
        
        # Check if we're dealing with datetime data
        has_datetime = False
        if x_param is not None and x_param in self.data.columns:
            if pd.api.types.is_datetime64_any_dtype(self.data[x_param]):
                has_datetime = True
        
        # Create main plot with left axis
        plot1 = win.addPlot()
        plot1.showGrid(x=True, y=True, alpha=0.3)
        
        # Set up date axis if we have datetime data
        if has_datetime:
            axis = pg.DateAxisItem(orientation='bottom')
            plot1.setAxisItems({'bottom': axis})
        
        # Set black color for axes, labels, and ticks on white background
        plot1.getAxis('bottom').setPen('k')
        plot1.getAxis('left').setPen('k')
        plot1.getAxis('bottom').setTextPen('k')
        plot1.getAxis('left').setTextPen('k')
        
        # Set x-axis label
        x_label = x_param if x_param is not None else "Index"
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
            plot1.getAxis('right').setPen('k')
            plot1.getAxis('right').setTextPen('k')
            
            # Add legends with borders and better positioning
            legend1 = plot1.addLegend(offset=(70, 10))
            legend1.setParentItem(plot1.graphicsItem())
            legend1.setBrush((255, 255, 255))
            legend1.setPen(pg.mkPen('k', width=2))
            legend1.setLabelTextSize('10pt')
            legend1.setZValue(100)
            
            legend2 = pg.LegendItem(offset=(-170, 10))
            legend2.setParentItem(plot1.graphicsItem())
            legend2.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-170, 10))
            legend2.setBrush((255, 255, 255))
            legend2.setPen(pg.mkPen('k', width=2))
            legend2.setLabelTextSize('10pt')
            legend2.setZValue(100)
        else:
            legend1 = plot1.addLegend(offset=(70, 10))
            legend1.setBrush((255, 255, 255))
            legend1.setPen(pg.mkPen('k', width=2))
            legend1.setLabelTextSize('10pt')
            legend1.setZValue(100)
        
        # Define colors (darker colors for white background)
        colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 128, 0), 
                (128, 0, 128), (0, 128, 128), (128, 0, 0), (0, 0, 128),
                (255, 0, 255), (0, 128, 0)]
        
        # Get x values
        if x_param is None:
            x_values = np.arange(len(self.data))
        else:
            # Check if x_param is datetime and convert to Unix timestamp
            if pd.api.types.is_datetime64_any_dtype(self.data[x_param]):
                x_values = self.data[x_param].astype(np.int64) / 10**9
            else:
                x_values = self.data[x_param]
        
        # Plot left axis parameters
        param_count = 0
        for y_param in left_params:
            y_values = self.data[y_param].values.copy()
            
            # Ensure x_values is a numpy array (convert from Series if needed)
            if isinstance(x_values, pd.Series):
                x_vals = x_values.values.copy()
            elif isinstance(x_values, np.ndarray):
                x_vals = x_values.copy()
            else:
                x_vals = np.array(x_values)
            
            # Convert to float arrays to ensure numeric operations work
            try:
                x_vals = np.asarray(x_vals, dtype=float)
                y_values = np.asarray(y_values, dtype=float)
            except (ValueError, TypeError):
                print(f'Warning: Could not convert {y_param} or {x_param} to numeric values. Skipping.')
                continue
            
            if sort_x and x_param is not None:
                sort_idx = np.argsort(x_vals)
                x_vals = x_vals[sort_idx]
                y_values = y_values[sort_idx]
            
            color = colors[param_count % len(colors)]
            
            # Create label for legend
            label = y_param if x_param is None else f'{y_param} vs {x_param}'
            
            # Add line and scatter plot to left axis
            line = pg.PlotDataItem(
                x=x_vals,
                y=y_values,
                pen=pg.mkPen(color, width=1.5),
                symbol=None
            )
            plot1.addItem(line)
            
            scatter = pg.ScatterPlotItem(
                x=x_vals, 
                y=y_values,
                pen=None,
                symbol='o',
                size=3,
                brush=color
            )
            plot1.addItem(scatter)
            
            legend1.addItem(line, label)
            param_count += 1
        
        # Plot right axis parameters if they exist
        if use_dual_axes and plot2 is not None:
            for y_param in right_params:
                y_values = self.data[y_param].values.copy()
                
                # Ensure x_values is a numpy array (convert from Series if needed)
                if isinstance(x_values, pd.Series):
                    x_vals = x_values.values.copy()
                elif isinstance(x_values, np.ndarray):
                    x_vals = x_values.copy()
                else:
                    x_vals = np.array(x_values)
                
                # Convert to float arrays to ensure numeric operations work
                try:
                    x_vals = np.asarray(x_vals, dtype=float)
                    y_values = np.asarray(y_values, dtype=float)
                except (ValueError, TypeError):
                    print(f'Warning: Could not convert {y_param} or {x_param} to numeric values. Skipping.')
                    continue
                
                if sort_x and x_param is not None:
                    sort_idx = np.argsort(x_vals)
                    x_vals = x_vals[sort_idx]
                    y_values = y_values[sort_idx]
                
                color = colors[param_count % len(colors)]
                
                # Create label for legend
                label = y_param if x_param is None else f'{y_param} vs {x_param}'
                
                # Add line and scatter plot to right axis
                line = pg.PlotDataItem(
                    x=x_vals,
                    y=y_values,
                    pen=pg.mkPen(color, width=1.5),
                    symbol=None
                )
                plot2.addItem(line)
                
                scatter = pg.ScatterPlotItem(
                    x=x_vals, 
                    y=y_values,
                    pen=None,
                    symbol='o',
                    size=3,
                    brush=color
                )
                plot2.addItem(scatter)
                
                legend2.addItem(line, label)
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
    
