import tkinter as tk
from tkinter import ttk
from Frame.DatasetFrame import DatasetFrame
from Frame.SettingFrame import SettingFrame

class App(tk.Tk):
  def __init__(self):
    super().__init__()
    self.initialize_GUI()
    
  def initialize_GUI(self):
    self.geometry('800x600')
    self.title('Pengenalan Wajah')
    self.notebook = ttk.Notebook(self)
    self.notebook.pack(expand=True, anchor="nw")
    
    self.dataset_frame = DatasetFrame(self.notebook)
    self.dataset_frame.pack(fill='both', expand=True)

    self.setting_frame = SettingFrame(self.notebook)
    self.setting_frame.pack(fill='both', expand=True)

    self.notebook.add(self.dataset_frame, text='Pengumpulan Dataset')
    self.notebook.add(self.setting_frame, text='Pengaturan')

if __name__ == "__main__":
  app = App()
  app.mainloop()