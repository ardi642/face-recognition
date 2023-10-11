import tkinter as tk
from tkinter import ttk

from configuration import configuration

class SettingFrame(tk.Frame):
  def __init__(self, master = None, **kwargs):
    super().__init__(master = None, **kwargs, pady=40, padx=40)
    self.master = master

    self.refresh_time = tk.IntVar()
    # self.refresh_time.set(configuration['refresh_time'])
    self.refresh_time.trace("w", self.update_refresh_time)

    self.refresh_time_label = tk.Label(self, text='waktu pembaruan frame')
    self.refresh_time_label.grid(column=0, row=1, padx=5, pady=5)

    self.refresh_time_entry = tk.Entry(self, textvariable=self.refresh_time)
    self.refresh_time_entry.grid(column=1, row=1, pady=5, padx=5)

    self.training_count = tk.IntVar()
    # self.training_count.set(configuration['training_image_count'])

    self.training_count_label = tk.Label(self, text='banyak gambar latih')
    self.training_count_label.grid(column=0, row=2, padx=5, pady=5)

    self.training_count_entry = tk.Entry(self, textvariable=self.training_count)
    self.training_count_entry.grid(column=1, row=2, pady=5, padx=5)

    self.training_count.trace("w", self.update_training_count)

    self.validation_count = tk.IntVar()

    self.additional_bottom = tk.IntVar()
    self.additional_bottom_label = tk.Label(self, text='penambahan bawah wajah')
    self.additional_bottom_label.grid(column=0, row=4, padx=5, pady=5)

    self.additional_bottom_entry = tk.Entry(self, textvariable=self.additional_bottom)
    self.additional_bottom_entry.grid(column=1, row=4, pady=5, padx=5)

    self.additional_bottom.trace("w", self.update_additional_bottom)

    self.reset_button = ttk.Button(self, text="Reset Pengaturan", 
    command=lambda: self.set_default_configuration(configuration=configuration))
    self.reset_button.grid(column=0, columnspan=2, row=5, padx=5, pady=5)

    self.set_default_configuration(configuration=configuration)

  def set_default_configuration(self, configuration):
    configuration["refresh_time"] = 10
    configuration["training_image_count"] = 30
    configuration["additional_bottom"] = 0

    self.refresh_time.set(configuration['refresh_time'])
    self.training_count.set(configuration['training_image_count'])
    self.additional_bottom.set(configuration['additional_bottom'])

  def update_refresh_time(self, *args):
    try:
      configuration['refresh_time'] = self.refresh_time.get()
    except tk.TclError:
      configuration['refresh_time'] = 0

  def update_training_count(self, *args):
    try:
      configuration['training_image_count'] = self.training_count.get()
    except tk.TclError:
      configuration['training_image_count'] = 0

  def update_additional_bottom(self, *args):
    try:
      configuration['additional_bottom'] = self.additional_bottom.get()
    except tk.TclError:
      configuration['additional_bottom'] = 0