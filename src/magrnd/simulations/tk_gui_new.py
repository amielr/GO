from tkinter import *
import tkinter.ttk as ttk
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from simulated_scan import SimulatedScan as Scan
from simulated_scan import MagneticDipole
from simulated_scan import ConstantField
from simulated_scan import FiniteDipoleChain
from simulated_scan import RectangularSource



class Main_window:
    def __init__(self):
        self.scan_class = Scan()
        self.master = Tk()
        buttonsframe = LabelFrame(self.master, text = 'Menu')
        buttonsframe.grid(row=0, column=0)
        self.fg = "#a6a6a6"
        self.bg = "#464646"

        self.input_text_1 = Label(buttonsframe, text='Location of scan(vector)').grid(row=0, column=0)
        self.button_addsource = Button(buttonsframe, text='Add source', command=self.open_window_add_source).grid(row=1,
                                                                                                                  column=0)  # i have to add action to the button
        self.button_changesource = Button(buttonsframe, text='Change source',command = self.open_window_changedelete_source).grid(row=2,
                                                                                   column=0)  # i have to add action to the button
        self.button_deletesource = Button(buttonsframe, text='Delete source').grid(row=3,
                                                                                   column=0)  # i have to add action to the button

        self.button_addscan = Button(buttonsframe, text='Add scan', command=self.open_window_add_route).grid(row=5,
                                                                                                             column=0)  # i have to add action to the button
        self.button_changescan = Button(buttonsframe, text='Change scan',command = self.open_window_changedelete_route).grid(row=6,
                                                                               column=0)  # i have to add action to the button
        self.button_deletescan = Button(buttonsframe, text='Delete scan').grid(row=7,
                                                                               column=0)  # i have to add action to the button

        self.save_button = Button(buttonsframe, text = 'Save GZ', command = self.scan_class.save).grid(row=8,
                                                                               column=0)

        self.display_button = Button(buttonsframe, text='Display', command=self.display_on_graph).grid(row=9,
                                                                                                       column=0)

        # table_frame = Frame(self.master, height= 350, width = 8)
        # table_frame.grid(row=10,column=0)
        # game_scroll = Scrollbar(table_frame)
        # game_scroll.grid(row=10, column=0)
        # game_scroll = Scrollbar(table_frame, orient = 'horizontal')
        # game_scroll.grid(row = 10, column = 0)
        # my_game = ttk.Treeview(game_scroll, yscrollcommand = game_scroll.set, xscrollcommand = game_scroll.set)

        # my_game.grid(row = 10, column = 0)


        self.fig = Figure(figsize=(6, 5), facecolor='white', linewidth=500, frameon=False)
        self.graphframe = LabelFrame(self.master, text = 'Simulation 3d plot')
        self.graphframe.grid(row=0, column=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graphframe)
        self.ax1 = self.fig.gca(projection='3d')
        self.ax1.set_xlabel('x axis')
        self.ax1.set_ylabel('y axis')
        self.ax1.set_zlabel('z axis')
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, ipadx=40, ipady=20)
        self.toolbar_frame = Frame(master=self.graphframe)
        self.toolbar_frame.grid(row=1, column=0)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.dipole_p_list = None
        self.dipole_m_list = None

        self.chain_p_init_list = None
        self.chain_p_final_list = None
        self.chain_m_list = None
        self.chain_density_list = None

        self.rect_p_init_list = None
        self.rect_p_final_list = None
        self.rect_density_list = None
        self.rect_m_list = None
        # self.startings values = when th window of the add source window is open

        self.var = IntVar(self.master)

        self.constantfield_class = ConstantField()

        self.spiral_route_list = None
        self.rect_route_list = None

        self.list_of_sources = []
        self.list_of_routes = []

        self.a_dipole = None
        self.a_chain = None

        self.n = 1


    # i have to add a window that provides information about the sources

    # self.scan  = Scan() #from their class
    def open_window_add_source(self):
        self.top = Toplevel()
        self.top.title('Add sources')
        self.var = IntVar()
        label1 = Label(self.top, text='Add external field', padx=80, pady=10).grid(row=0, column=0)
        self.yesbutton = Radiobutton(self.top, text='YES', variable=self.var, value=1).grid(row=1, column=0)
        self.nobutton = Radiobutton(self.top, text='NO', variable=self.var, value=0).grid(row=2, column=0)
        input_text1 = Label(self.top, text='Single source:').grid(row=4, column=0)
        input_text2 = Label(self.top, text='Chain source:').grid(row=6, column=0)
        input_text3 = Label(self.top, text='Rectangular source:').grid(row=10, column=0)

        # dipole
        Label(self.top, text='x').grid(row=3, column=1)
        Label(self.top, text='y').grid(row=3, column=2)
        Label(self.top, text='z').grid(row=3, column=3)
        Label(self.top, text='mx').grid(row=3, column=4)
        Label(self.top, text='my').grid(row=3, column=5)
        Label(self.top, text='mz').grid(row=3, column=6)

        self.entry_dipole_1 = Entry(self.top, width=4)
        self.entry_dipole_1.grid(row=4, column=1)
        self.entry_dipole_2 = Entry(self.top, width=4)
        self.entry_dipole_2.grid(row=4, column=2)
        self.entry_dipole_3 = Entry(self.top, width=4)
        self.entry_dipole_3.grid(row=4, column=3)
        self.entry_dipole_4 = Entry(self.top, width=4)
        self.entry_dipole_4.grid(row=4, column=4)
        self.entry_dipole_5 = Entry(self.top, width=4)
        self.entry_dipole_5.grid(row=4, column=5)
        self.entry_dipole_6 = Entry(self.top, width=4)
        self.entry_dipole_6.grid(row=4, column=6)

        # chain
        Label(self.top, text='x initial').grid(row=5, column=1)
        Label(self.top, text='y initial').grid(row=5, column=2)
        Label(self.top, text='z initial').grid(row=5, column=3)
        Label(self.top, text='mx').grid(row=5, column=4)
        Label(self.top, text='my').grid(row=5, column=5)
        Label(self.top, text='mz').grid(row=5, column=6)

        self.entry_chain_1 = Entry(self.top, width=4)
        self.entry_chain_1.grid(row=6, column=1)
        self.entry_chain_2 = Entry(self.top, width=4)
        self.entry_chain_2.grid(row=6, column=2)
        self.entry_chain_3 = Entry(self.top, width=4)
        self.entry_chain_3.grid(row=6, column=3)
        self.entry_chain_4 = Entry(self.top, width=4)
        self.entry_chain_4.grid(row=6, column=4)
        self.entry_chain_5 = Entry(self.top, width=4)
        self.entry_chain_5.grid(row=6, column=5)
        self.entry_chain_6 = Entry(self.top, width=4)
        self.entry_chain_6.grid(row=6, column=6)

        Label(self.top, text='x final').grid(row=7, column=1)
        Label(self.top, text='y final').grid(row=7, column=2)
        Label(self.top, text='z final').grid(row=7, column=3)
        Label(self.top, text='density').grid(row=7, column=4)

        self.entry_chain_7 = Entry(self.top, width=4)
        self.entry_chain_7.grid(row=8, column=1)
        self.entry_chain_8 = Entry(self.top, width=4)
        self.entry_chain_8.grid(row=8, column=2)
        self.entry_chain_9 = Entry(self.top, width=4)
        self.entry_chain_9.grid(row=8, column=3)
        self.entry_chain_10 = Entry(self.top, width=4)
        self.entry_chain_10.grid(row=8, column=4)

        # rectangular
        Label(self.top, text='x initial').grid(row=9, column=1)
        Label(self.top, text='y initial').grid(row=9, column=2)
        Label(self.top, text='z initial').grid(row=9, column=3)
        Label(self.top, text='mx').grid(row=9, column=4)
        Label(self.top, text='my').grid(row=9, column=5)
        Label(self.top, text='mz').grid(row=9, column=6)

        self.entry_rect_1 = Entry(self.top, width=4)
        self.entry_rect_1.grid(row=10, column=1)
        self.entry_rect_2 = Entry(self.top, width=4)
        self.entry_rect_2.grid(row=10, column=2)
        self.entry_rect_3 = Entry(self.top, width=4)
        self.entry_rect_3.grid(row=10, column=3)
        self.entry_rect_4 = Entry(self.top, width=4)
        self.entry_rect_4.grid(row=10, column=4)
        self.entry_rect_5 = Entry(self.top, width=4)
        self.entry_rect_5.grid(row=10, column=5)
        self.entry_rect_6 = Entry(self.top, width=4)
        self.entry_rect_6.grid(row=10, column=6)

        Label(self.top, text='x final').grid(row=11, column=1)
        Label(self.top, text='y final').grid(row=11, column=2)
        Label(self.top, text='z final').grid(row=11, column=3)
        Label(self.top, text='x density').grid(row=11, column=4)
        Label(self.top, text='y density').grid(row=11, column=5)

        self.entry_rect_7 = Entry(self.top, width=4)
        self.entry_rect_7.grid(row=12, column=1)
        self.entry_rect_8 = Entry(self.top, width=4)
        self.entry_rect_8.grid(row=12, column=2)
        self.entry_rect_9 = Entry(self.top, width=4)
        self.entry_rect_9.grid(row=12, column=3)
        self.entry_rect_10 = Entry(self.top, width=4)
        self.entry_rect_10.grid(row=12, column=4)
        self.entry_rect_11 = Entry(self.top, width=4)
        self.entry_rect_11.grid(row=12, column=5)

        apply_button = Button(self.top, text='Apply', command=self.add_sources).grid(row=13, column=1)

    def add_sources(self):
        '''
        adding constant field of the earth
        '''
        if self.var.get() == 1:
            B_earth1 = ConstantField()
            self.scan_class.add_source(B_earth1)
        else:
            pass
        '''
        adding one dipole 
        '''
        if self.dipole_p_list is None or self.dipole_m_list is None:
            self.dipole_p_list = {'x': [], 'y': [], 'z': []}
            self.dipole_m_list = {'mx': [], 'my': [], 'mz': []}

        if self.chain_p_init_list is None:
            self.chain_p_init_list = {'x': [], 'y': [], 'z': []}
            self.chain_p_final_list = {'x': [], 'y': [], 'z': []}
            self.chain_m_list = {'mx': [], 'my': [], 'mz': []}
            self.chain_density_list = {'d': []}

        if self.rect_m_list is None:
            self.rect_p_init_list = {'x': [], 'y': [], 'z': []}
            self.rect_p_final_list = {'x': [], 'y': [], 'z': []}
            self.rect_density_list = {'dx': [],'dy': []}
            self.rect_m_list = {'mx': [], 'my': [], 'mz': []}
        try:
            self.dipole_p_list['x'].append(float(self.entry_dipole_1.get()))
            self.dipole_p_list['y'].append(float(self.entry_dipole_2.get()))
            self.dipole_p_list['z'].append(float(self.entry_dipole_3.get()))
            self.dipole_m_list['mx'].append(float(self.entry_dipole_4.get()))
            self.dipole_m_list['my'].append(float(self.entry_dipole_5.get()))
            self.dipole_m_list['mz'].append(float(self.entry_dipole_6.get()))
            self.ax1.scatter3D(self.dipole_p_list['x'][-1], self.dipole_p_list['y'][-1], self.dipole_p_list['z'][-1],
                               label='Source {}-one dipole'.format(self.n))
            self.magnetic_dipole = MagneticDipole(
                p_vec=[self.dipole_p_list['x'][-1], self.dipole_p_list['y'][-1], self.dipole_p_list['z'][-1]],
                m_vec=[self.dipole_m_list['mx'][-1], self.dipole_m_list['mx'][-1], self.dipole_m_list['mx'][-1]])
            self.scan_class.add_source(self.magnetic_dipole)
            self.ax1.legend()
            self.n += 1
        except ValueError:
            pass

        '''
        adding chain
        '''
        try:
            self.chain_p_init_list['x'].append(float(self.entry_chain_1.get()))
            self.chain_p_init_list['y'].append(float(self.entry_chain_2.get()))
            self.chain_p_init_list['z'].append(float(self.entry_chain_3.get()))
            self.chain_p_final_list['x'].append(float(self.entry_chain_7.get()))
            self.chain_p_final_list['y'].append(float(self.entry_chain_8.get()))
            self.chain_p_final_list['z'].append(float(self.entry_chain_9.get()))
            self.chain_m_list['mx'].append(float(self.entry_chain_4.get()))
            self.chain_m_list['my'].append(float(self.entry_chain_5.get()))
            self.chain_m_list['mz'].append(float(self.entry_chain_6.get()))
            self.chain_density_list['d'].append(float(self.entry_chain_10.get()))
            print(self.n)
            self.magnetic_chain = FiniteDipoleChain(
                p_initial =np.array([self.chain_p_init_list['x'][-1],self.chain_p_init_list['y'][-1],self.chain_p_init_list['z'][-1]]),
                p_final =np.array([self.chain_p_final_list['x'][-1],self.chain_p_final_list['y'][-1],self.chain_p_final_list['z'][-1]]),
                m = np.array([self.chain_m_list['mx'][-1],self.chain_m_list['my'][-1],self.chain_m_list['mz'][-1]]),
                density = float(self.chain_density_list['d'][-1]))
            self.scan_class.add_source(self.magnetic_chain)
            self.chain_source_last = self.scan_class.scan_sources[-1].get_pos()
            self.ax1.scatter3D(self.chain_source_last[:,0], self.chain_source_last[:,1], self.chain_source_last[:,2],
                               label='Source {}-chain'.format(self.n))
            self.ax1.legend()
            self.n += 1


        except ValueError:
         pass
         '''
         adding rectangular source
         '''
        try:
            self.rect_p_init_list['x'].append(float(self.entry_rect_1.get()))
            self.rect_p_init_list['y'].append(float(self.entry_rect_1.get()))
            self.rect_p_init_list['z'].append(float(self.entry_rect_3.get()))
            self.rect_p_final_list['x'].append(float(self.entry_rect_7.get()))
            self.rect_p_final_list['y'].append(float(self.entry_rect_8.get()))
            self.rect_p_final_list['z'].append(float(self.entry_rect_9.get()))
            self.rect_m_list['mx'].append(float(self.entry_rect_4.get()))
            self.rect_m_list['my'].append(float(self.entry_rect_5.get()))
            self.rect_m_list['mz'].append(float(self.entry_rect_6.get()))
            self.rect_density_list['dx'].append(float(self.entry_rect_10.get()))
            self.rect_density_list['dy'].append(float(self.entry_rect_10.get()))

            self.magnetic_rect = RectangularSource(
                p_min =np.array([self.rect_p_init_list['x'][-1],self.rect_p_init_list['y'][-1],self.rect_p_init_list['z'][-1]]),
                p_max =np.array([self.rect_p_final_list['x'][-1],self.rect_p_final_list['y'][-1],self.rect_p_final_list['z'][-1]]),
                m = np.array([self.rect_m_list['mx'][-1],self.rect_m_list['my'][-1],self.rect_m_list['mz'][-1]]),
                x_density = float(self.rect_density_list['dx'][-1]), y_density = float(self.rect_density_list['dy'][-1]))
            self.scan_class.add_source(self.magnetic_rect)
            self.rect_source_last = self.scan_class.scan_sources[-1].get_pos()
            self.ax1.scatter3D(self.rect_source_last[:,0], self.rect_source_last[:,1], self.rect_source_last[:,2],
                               label='Source {}-rectangular'.format(self.n))
            self.ax1.legend()
            self.n += 1
        except ValueError:
            pass


        #self.scan_class.add_spiral_route(a=2, b=1, x0 = 7, y0 = 8, z0 = 20)
        #self.display_on_graph()
        self.canvas.draw()


        self.top.destroy()

    def open_window_add_route(self):
        self.top1 = Toplevel()
        self.top1.title('Add route')
        self.var = IntVar()
        Label(self.top1, text='Add route from external file:', padx=80, pady=10).grid(row=0, column=0)
        Button(self.top1, text='Open file',command = self.add_route_from_external_file).grid(row=1,column=0)
        Label(self.top1, text='Add route manually:', padx=80, pady=10).grid(row=2, column=0)
        Label(self.top1, text='spiral route:', padx=80, pady=10).grid(row=3, column=0)
        Label(self.top1, text = 'rectangular route:', padx = 80, pady = 10).grid(row = 5, column = 0)

        Label(self.top1, text='   a   ').grid(row=3, column=1)
        Label(self.top1, text='   b   ').grid(row=3, column=2)
        Label(self.top1, text='n\n(number of spirals)').grid(row=3, column=3)
        Label(self.top1, text='x0 \n (x coord of the center)').grid(row=3, column=4)
        Label(self.top1, text='y0\n(y coord of the center)').grid(row=3, column=5)
        Label(self.top1, text='z0\n(height)').grid(row=3, column=6)
        Label(self.top1, text='sample rate').grid(row=3, column=7)

        self.entry_spiral_1 = Entry(self.top1, width=4)
        self.entry_spiral_1.grid(row=4, column=1)
        self.entry_spiral_2 = Entry(self.top1, width=4)
        self.entry_spiral_2.grid(row=4, column=2)
        self.entry_spiral_3 = Entry(self.top1, width=4)
        self.entry_spiral_3.grid(row=4, column=3)
        self.entry_spiral_4 = Entry(self.top1, width=4)
        self.entry_spiral_4.grid(row=4, column=4)
        self.entry_spiral_5 = Entry(self.top1, width=4)
        self.entry_spiral_5.grid(row=4, column=5)
        self.entry_spiral_6 = Entry(self.top1, width=4)
        self.entry_spiral_6.grid(row=4, column=6)
        self.entry_spiral_7 = Entry(self.top1, width=4)
        self.entry_spiral_7.grid(row=4, column=7)

        Label(self.top1, text='x0').grid(row=5, column=1)
        Label(self.top1, text='x1').grid(row=7, column=1)
        Label(self.top1, text='y0').grid(row=5, column=2)
        Label(self.top1, text='y1').grid(row=7, column=2)
        Label(self.top1, text='z').grid(row=9, column=1)
        Label(self.top1, text='fs').grid(row=5, column=3)
        Label(self.top1, text='v').grid(row=7, column=3)
        Label(self.top1, text='number of\n lines').grid(row=5, column=4)

        self.entry_rectroute_1 = Entry(self.top1, width=4)
        self.entry_rectroute_1.grid(row=6, column=1)
        self.entry_rectroute_2 = Entry(self.top1, width=4)
        self.entry_rectroute_2.grid(row=8, column=1)
        self.entry_rectroute_3 = Entry(self.top1, width=4)
        self.entry_rectroute_3.grid(row=6, column=2)
        self.entry_rectroute_4 = Entry(self.top1, width=4)
        self.entry_rectroute_4.grid(row=8, column=2)
        self.entry_rectroute_5 = Entry(self.top1, width=4)
        self.entry_rectroute_5.grid(row=10, column=1)
        self.entry_rectroute_6 = Entry(self.top1, width=4)
        self.entry_rectroute_6.grid(row=6, column=3)
        self.entry_rectroute_7 = Entry(self.top1, width=4)
        self.entry_rectroute_7.grid(row=8, column=3)
        self.entry_rectroute_8 = Entry(self.top1, width=4)
        self.entry_rectroute_8.grid(row=6, column=4)

        Button(self.top1, text='Apply', command=self.add_routes).grid(row=11, column=1)
    def add_route_from_external_file(self):
        self.scan_class.add_route_from_gz_file()
        #not working now
    def add_routes(self):
        '''

        create spiral route

        '''

        if self.spiral_route_list is None:
            self.spiral_route_list = {'a': [], 'b': [], 'n': [], 'x0': [], 'y0': [], 'z0': [], 'sample rate': []}

        if self.rect_route_list is None:
            self.rect_route_list = {'x0': [], 'y0': [], 'x1': [], 'y1': [], 'v': [], 'z': [], 'fs': [], 'lines': []}
        try:
            self.spiral_route_list['a'].append(int(self.entry_spiral_1.get()))
            self.spiral_route_list['b'].append(int(self.entry_spiral_2.get()))
            self.spiral_route_list['n'].append(int(self.entry_spiral_3.get()))
            self.spiral_route_list['x0'].append(int(self.entry_spiral_4.get()))
            self.spiral_route_list['y0'].append(int(self.entry_spiral_5.get()))
            self.spiral_route_list['z0'].append(int(self.entry_spiral_6.get()))
            self.spiral_route_list['sample rate'].append(int(self.entry_spiral_7.get()))
            self.scan_class.add_spiral_route(a = self.spiral_route_list['a'][-1], b = self.spiral_route_list['b'][-1],
                                             x0 =self.spiral_route_list['x0'][-1], y0 = self.spiral_route_list['y0'][-1],
                                             n = self.spiral_route_list['n'][-1],
                                             z0 =self.spiral_route_list['z0'][-1])
                                             # sample_num =self.spiral_route_list['sample rate'][-1])

            self.spiral_route_last = self.scan_class.scan_route[-1]
            self.ax1.plot(self.spiral_route_last[:, 0], self.spiral_route_last[:, 1],
                          self.spiral_route_last[:, 2],label='Route {}-Spiral'.format(len(self.spiral_route_list['x0'])+
                                                                                      len(self.rect_route_list['x0'])))
            # self.ax3.plot(self.rect_scan_last[:, 0], self.rect_scan_last[:, 1], self.rect_scan_last[:, 2])
            self.ax1.legend()
            self.canvas.draw()
        except ValueError:
            pass
        '''

        create rectangular route

        '''
        try:
            self.rect_route_list['x0'].append(int(self.entry_rectroute_1.get()))
            self.rect_route_list['x1'].append(int(self.entry_rectroute_2.get()))
            self.rect_route_list['y0'].append(int(self.entry_rectroute_3.get()))
            self.rect_route_list['y1'].append(int(self.entry_rectroute_4.get()))
            self.rect_route_list['z'].append(int(self.entry_rectroute_5.get()))
            self.rect_route_list['fs'].append(int(self.entry_rectroute_6.get()))
            self.rect_route_list['v'].append(int(self.entry_rectroute_7.get()))
            self.rect_route_list['lines'].append(int(self.entry_rectroute_8.get()))

            self.scan_class.add_rectangular_route(x0 = self.rect_route_list['x0'][-1],y0 = self.rect_route_list['y0'][-1],
                                                  x1 = self.rect_route_list['x1'][-1],y1 = self.rect_route_list['y1'][-1],
                                                  z = self.rect_route_list['z'][-1],fs = self.rect_route_list['fs'][-1],
                                                  v = self.rect_route_list['v'][-1],num_lines = self.rect_route_list['lines'][-1])
            self.rect_scan_last = self.scan_class.scan_route[-1]
            self.ax1.plot(self.rect_scan_last[:, 0], self.rect_scan_last[:, 1], self.rect_scan_last[:, 2],
                          label = 'Route {}-Rectangular'.format(len(self.rect_route_list['x0'])+len(self.spiral_route_list['x0'])))
            #self.ax3.plot(self.rect_scan_last[:, 0], self.rect_scan_last[:, 1], self.rect_scan_last[:, 2])
            self.ax1.legend()
            self.canvas.draw()
        except ValueError:
            pass

        self.top1.destroy()

    def open_window_changedelete_source(self):
        self.top3 = Tk()
        buttonsframe2 = LabelFrame(self.top3, text = 'Add & Remove sources')
        buttonsframe2.grid(row=0, column=0)
        Label(buttonsframe2, text='Select source:').grid(row=0, column=0)
        j = 1
        for source in self.scan_class.scan_sources:
            if not isinstance(source, ConstantField):
                self.list_of_sources.append('Source {}'.format(j))
                j += 1
        combo_button_sources = ttk.Combobox(buttonsframe2, values=self.list_of_sources)
        combo_button_sources.grid(row=1, column=0)

        self.list_of_sources = []
        self.fig1 = Figure(figsize=(5, 4), facecolor='white', linewidth=500, frameon=False)
        self.graphframe2 = LabelFrame(self.top3, text='Sources')
        self.graphframe2.grid(row=0, column=1)
        self.canvas2 = FigureCanvasTkAgg(self.fig1, master=self.graphframe2)
        self.ax2 = self.fig1.gca(projection='3d')
        self.ax2.set_xlabel('x axis')
        self.ax2.set_ylabel('y axis')
        self.ax2.set_zlabel('z axis')
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(row=0, column=0, ipadx=40, ipady=20)
        self.toolbar_frame2 = Frame(master=self.graphframe2)
        self.toolbar_frame2.grid(row=1, column=0)
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.toolbar_frame2)
        self.toolbar2.update()
        i = 1
        for source in self.scan_class.scan_sources:
            if not isinstance(source, ConstantField):
                if np.array(source.get_pos()).shape == (3,):
                    pos = np.array([source.get_pos()])
                else:
                    pos = source.get_pos()

                self.ax2.scatter3D(pos[:, 0], pos[:, 1], pos[:, 2], label = 'Source {}'.format(i))
                i += 1



        #if combo_button_sources.get() == 'Source 1-one dipole':
            print('ffff')

        # #chain
        # try:
        #     for m in range(self.a_dipole,len(self.dipole_p_list['x'])+2):
        # self.ax2.scatter3D(self.dipole_p_list['x'][i - 1], self.dipole_p_list['y'][i - 1],
        #                    self.dipole_p_list['z'][i - 1],
        #                    label='Source {}-chain'.format(i))
        self.ax2.legend()
        self.canvas2.draw()
        Button(self.top3, text='Apply', command=self.changedelete_source_apply).grid(row=11, column=1)
    def changedelete_source_apply(self):

        self.top3.destroy()


        pass
    def open_window_changedelete_route(self):
        self.top4 = Tk()
        buttonsframe4 = LabelFrame(self.top4, text='Add & Remove routes')
        buttonsframe4.grid(row=0, column=0)
        Label(buttonsframe4, text='Select route:').grid(row=0, column=0)
        try:
            for i in range( 1,len(self.dipole_p_list['x'])+1):
                self.list_of_sources.append('Source {}-one dipole'.format(i))
                self.a_dipole = i+1
        except (ValueError,TypeError) as e:
            pass
        for route in self.scan_class.scan_route:
            self.ax3.plot(route[:, 0], route[:, 1], route[:, 2], label='route {}'.format(self.r_s))
            self.r_s += 1

        combo_button_routes = ttk.Combobox(buttonsframe4, values=self.list_of_routes)
        combo_button_routes.grid(row=1, column=0)

        self.list_of_routes = []
        self.fig2 = Figure(figsize=(5, 4), facecolor='white', linewidth=500, frameon=False)
        self.graphframe4 = LabelFrame(self.top4, text='Sources')
        self.graphframe4.grid(row=0, column=1)
        self.canvas3 = FigureCanvasTkAgg(self.fig2, master=self.graphframe4)
        self.ax3 = self.fig2.gca(projection='3d')
        self.ax3.set_xlabel('x axis')
        self.ax3.set_ylabel('y axis')
        self.ax3.set_zlabel('z axis')
        self.canvas3.draw()
        self.canvas3.get_tk_widget().grid(row=0, column=0, ipadx=40, ipady=20)
        self.toolbar_frame3 = Frame(master=self.graphframe4)
        self.toolbar_frame3.grid(row=1, column=0)
        self.toolbar3 = NavigationToolbar2Tk(self.canvas3, self.toolbar_frame3)
        self.toolbar3.update()
        self.r_s = 1
        try:
            for route in self.scan_class.scan_route:
                self.ax3.plot(route[:, 0], route[:, 1], route[:, 2], label = 'route {}'.format(self.r_s))
                self.r_s +=1
        except TypeError:
            pass
        self.ax3.legend()
        self.canvas3.draw()


    def display_on_graph(self, display_scan=True,display_sources=True, display_magnetic_field=True):
        try:
            z_min = [min(self.scan_class.get_joint_route()[:, 2]) - 5]
            z_max = [max(self.scan_class.get_joint_route()[:, 2]) + 5]

            x_min = [min(self.scan_class.get_joint_route()[:, 0]) - 5]
            x_max = [max(self.scan_class.get_joint_route()[:, 0]) + 5]

            y_min = [min(self.scan_class.get_joint_route()[:, 1]) - 5]
            y_max = [max(self.scan_class.get_joint_route()[:, 1]) + 5]
            if display_sources:
                for source in self.scan_class.scan_sources:
                    if not isinstance(source.get_pos(), ConstantField):
                        if np.array(source.get_pos()).shape == (3,):
                            pos = np.array([source.get_pos()])
                        else:
                            pos = source.get_pos()
                            z_min.append(min(pos[:, 2] - 5))
                            z_max.append(max(pos[:, 2] + 5))

                            x_min.append(min(pos[:, 0]) - 5)
                            x_max.append(max(pos[:, 0]) + 5)

                            y_min.append(min(pos[:, 1]) - 5)
                            y_max.append(max(pos[:, 1]) + 5)

            if display_magnetic_field:
                B_whole = self.scan_class.calculate_magnetic_field()
                v_min = min(B_whole) * 10 ** 9
                v_max = max(B_whole) * 10 ** 9
                for i in range(len(self.scan_class.scan_route)):
                    B = self.scan_class.calculate_magnetic_field(route_wise=True)[i]
                    self.ax1.tricontourf(self.scan_class.scan_route[i][:, 0], self.scan_class.scan_route[i][:, 1],
                                         B * 10 ** 9, zdir='z',
                                         offset=np.average(self.scan_class.scan_route[i][:, 2]) - 1,
                                         vmin=v_min, vmax=v_max, cmap='jet', levels=30)
            _min = np.min([x_min, y_min])
            _max = np.max([x_max, y_max])
            self.ax1.set_zlim(min(z_min), max(z_max))
            self.ax1.set_xlim(min(x_min), max(x_max))
            self.ax1.set_ylim(min(y_min), max(y_max))

            self.ax1.set_xlabel('East [m]')
            self.ax1.set_ylabel('North [m]')
            self.ax1.set_zlabel('Up [m]')
            self.canvas.draw()
        except ValueError:
            pass


if __name__ == "__main__":
    window = Main_window()
    window.master.mainloop()
