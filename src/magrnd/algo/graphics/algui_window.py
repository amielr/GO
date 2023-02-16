import codecs
import json
from tkinter.filedialog import askopenfilename
from ttkthemes import ThemedTk
from pathlib import Path
from mag_utils.mag_utils.scans.horizontal_scan import HorizontalScan
from magrnd.algo.consts import GUI_THEME, WINDOW_TITLE, FILE_NAME_FONT, TITLE_FONT
from magrnd.algo.graphics.tabs import ALGORITHM_TAB_LIST
from tkinter.ttk import Label, Notebook
from tkinter import Menu
import matplotlib.pyplot as plt
from magrnd.ground_one.loaders.mag_loader import load


class Algui:
    def __init__(self, scan: HorizontalScan):
        self.scan = scan
        self.keyboard_shortcuts = {"s": self.save,
                                   "o": self.load}

        # init window
        self.initialize_window()

        # display window
        self.root.mainloop()

    def initialize_window(self):
        # create window
        self.root = ThemedTk(themebg=True)
        self.root.set_theme(GUI_THEME)
        self.root.wm_title(WINDOW_TITLE)

        self.window_title = Label(master=self.root, text=WINDOW_TITLE, font=TITLE_FONT)
        self.window_title.pack(side='top')
        self.scan_title = Label(master=self.root, text=Path(self.scan.file_name).stem, font=FILE_NAME_FONT)
        self.scan_title.pack(side='top')

        main_menu = Menu(self.root)
        self.root.config(menu=main_menu)
        file_menu = Menu(main_menu)

        # add file menu
        main_menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_cascade(label='Load', command=self.load)
        file_menu.add_cascade(label='Save', command=self.save)

        # add keyboard shortcuts
        plt.rcParams['keymap.save'].remove("ctrl+s")
        self.root.bind("<Key>", lambda event: self.handle_keypress(event=event))

        # creating the tab control
        self.tab_control = Notebook(master=self.root)
        self.tabs = [algo_tab(tab_control=self.tab_control, scan=self.scan) for algo_tab in ALGORITHM_TAB_LIST]
        self.tab_control.pack(expand=1, fill='both')

    def get_current_tab(self):
        selected_index = self.tab_control.index(self.tab_control.select())
        return self.tabs[selected_index]

    def save(self):
        tab = self.get_current_tab()
        tab.save()

    def load(self):
        file_path = Path(askopenfilename(filetypes=(("All files", "*.*"),)))

        # get current tab
        current_tab = self.get_current_tab()

        # if scan was selected
        if file_path.name.lower().endswith(".txt"):
            scan = load(file_path)

            for tab in self.tabs:
                tab.set_scan(scan)

            self.scan_title.config(text=Path(scan.file_name).stem)

        # else if a ALGO JSON file was selected
        elif file_path.name.lower().endswith(".json"):
            with codecs.open(str(file_path), mode="rb", encoding="utf-8") as file:
                file_dict = json.load(file)

            # switch to relevant algorithm tab
            tab_names = [self.tab_control.tab(tab_id, "text") for tab_id in self.tab_control.tabs()]
            algorithm_index = tab_names.index(file_dict["Metadata"]["Algorithm"])
            self.tab_control.select(algorithm_index)

            scan = load(Path(file_dict["Metadata"]["Path"]))
            for tab in self.tabs:
                tab.set_scan(scan)

            # load p[arameters and results
            self.tabs[algorithm_index].load_parameters_and_results(file_dict)

            # update scan title
            self.scan_title.config(text=Path(file_dict["Metadata"]["Path"]).stem)
        else:
            raise Exception("File type is not supported")

    def handle_keypress(self, event):
        if event.state & 4 <= 0:
            return

        key_pressed = str(chr(event.keycode)).lower()

        if key_pressed in self.keyboard_shortcuts:
            shortcut_callback_func = self.keyboard_shortcuts[key_pressed]
            shortcut_callback_func()
