import tkinter as tk


class ScanTagGui:
    def __init__(self):
        self.tag = None
        self.window = tk.Tk()
        self.window.geometry('300x150+100+100')
        self.window.title('tag your scan')

    def ia_pressed(self):
        self.tag = 'ia'
        self.window.destroy()

    def widow_pressed(self):
        self.tag = 'widow'
        self.window.destroy()

    def build_window(self):
        instruction_text = tk.Label(self.window,
                                         text="choose your scan type",
                                         width=20, height=3, font=('David', 20))
        instruction_text.place(x=0, y=0)

        ia_button = tk.Button(master=self.window, text="IA", width=15, height=3, command=self.ia_pressed)
        ia_button.place(x=20, y=80)
        widow_button = tk.Button(master=self.window, text="widow", width=15, height=3, command=self.widow_pressed)
        widow_button.place(x=170, y=80)

        tk.mainloop()

if __name__ == '__main__':
    obj = ScanTagGui()
    obj.build_window()
    print(obj.tag)
