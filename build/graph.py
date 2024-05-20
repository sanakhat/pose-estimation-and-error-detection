import pandas as pd
import matplotlib.pyplot as plt
from pandastable import Table
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class CSVReaderApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.title("CSV Reader")

        self.table_frame = tk.Frame(self)
        self.table_frame.pack(pady=10)

        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.graph = FigureCanvasTkAgg(self.fig, self.canvas)
        self.graph.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.graph, self.canvas)
        self.toolbar.update()
        self.graph._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.bind("<Button-1>", self.on_click)

        self.read_csv_and_plot()

    def display_table(self, dataframe):
        if hasattr(self, "table"):
            self.table.destroy()
        self.table = Table(self.table_frame, dataframe=dataframe)
        self.table.show()

    def read_csv_and_plot(self):
        file_path = 'C:/Users/Admin/OneDrive/Desktop/build/error_rates.csv'  # Specify file path directly
        df = pd.read_csv(file_path)
        
        self.display_table(df)
        
        # Extracting columns for plotting
        self.Total_Error = df['Total Error']
        self.Valid_Pairs = df['Valid Pairs']
        
        # Plotting the graph
        self.plot.clear()
        self.plot.plot(self.Total_Error, self.Valid_Pairs)
        self.plot.set_title('Total Error vs Valid Pairs')
        self.plot.set_xlabel('Total Error')
        self.plot.set_ylabel('Valid Pairs')
        self.plot.grid(True)
        self.graph.draw()

    def on_click(self, event):
        if event.inaxes == self.plot:
            x, y = event.xdata, event.ydata
            nearest_index = (abs(self.Total_Error - x) + abs(self.Valid_Pairs - y)).idxmin()
            self.table.show_index(nearest_index)

app = CSVReaderApp()
app.mainloop()
