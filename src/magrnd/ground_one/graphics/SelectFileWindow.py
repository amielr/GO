from tkinter import mainloop
from magrnd.ground_one.data_processing.consts import GUI_THEME, SELECT_FILE_WINDOW_FONT
from ttkthemes import ThemedTk
from tkinter.ttk import Label, Button
from magrnd.ground_one.loaders.mag_loader import load


class SelectFileWindow:
    def __init__(self):
        self.window = ThemedTk(themebg=True)
        self.window.set_theme(GUI_THEME)

        self.window.wm_title("Select File Window")

        label = Label(self.window, text="Choose the file you want to work with:")
        label.config(font=SELECT_FILE_WINDOW_FONT)
        label.grid(row=0, column=1)

        choose_file_button = Button(master=self.window, text="Choose file", command=self.file_loader)
        choose_file_button.grid(row=1, column=1, ipady=20, ipadx=20)
        self.window.bind("<Return>", func=lambda x: self.file_loader())
        mainloop()  # Ends the gui

    def file_loader(self):
        self.scans = load()
        self.window.quit()
        self.window.destroy()

    def get_scan(self):
        return self.scans


if __name__ == "__main__":
    win = SelectFileWindow()
    print(win.get_scan())
