import tkinter as tk
from tkinter import filedialog
import pathlib as pth
from typing import Optional, Union
import sys

def pick_file(title: str = "Choose a data file",
                initial_dir: Optional[Union[str, pth.Path]] = None, 
                filetypes: tuple = (("CSV Files", "*.csv"),
                                    ("Excel Files", ".xlsx"),
                                    ("All Files", "*.*")
                                    )) -> Optional[pth.Path]:
    """
    Opens a Tkinter file selection dialog and returns the chosen path as a pathlib.Path object.

    Args:
        title (str): The title for the file dialog window.
        initial_dir (Optional[Path]): Starting directory. Defaults to the user's home directory.
        filetypes (tuple): File type filter (e.g., (("Data files", "*.dat"),)). CSV is now the default.

    Returns:
        Optional[Path]: The selected file path (pathlib.Path) or None if cancelled.
    """
    
    # Initialize and hide the root window for clean dialog pop-up
    root = tk.Tk()
    root.withdraw() 
    
    # Determine start directory
    if initial_dir is None:
        start_dir = pth.Path(__file__).parent.parent.joinpath('data')
    elif isinstance(initial_dir, str):
        start_dir = Path(initial_dir)
    else:
        start_dir = initial_dir

    # Open the file selection dialog
    output_path = None
    try:
        output_path = filedialog.askopenfilename(
            title=title,
            initialdir=str(start_dir),
            filetypes=filetypes
        )
    except Exception as e:
        print(e)
        output_path = None

    # Clean up the hidden root window
    root.destroy()

    # Convert string path to pathlib.Path or return None
    if output_path:
        output_path = pth.Path(output_path)
        return output_path
    else:
        print('Window closed and selection has been cancelled. Exiting...')
        sys.exit()

def test_pick_file():
    # Example usage
    selected_path = pick_file(
        title="Choose a Data File",
        initial_dir=pth.Path(__file__).parent.parent.joinpath('data'),
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )

    if selected_path:
        print(f"\nPath selected: {selected_path}")
    else:
        print("\nSelection cancelled.")

if __name__ == '__main__':
    test_pick_file()