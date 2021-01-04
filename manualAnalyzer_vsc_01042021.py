# -*- coding: utf-8 -*-
"""
Manual Hodograph Analyzer
Include this script in the same folder as the input and output directories, or else specify their file path

Semantics:
Enclosed structures contained in full-flight hodographs are reffered to as "microhodographs", 

To Do:
- add/update sources for parameter calculations
- format parameter output to match wavelet python code? Maybe doesnt make sense to use json, as profile data needs to accompany wave parameters for plotting
- get rid of metpy library?
- clean up bulk hodograph plots
- add time to wave parameter list
    -

Malachi Mooney-Rivkin
Last Edit: 12/29/2020
Idaho Space Grant Consortium
moon8435@vandals.uidaho.edu
"""

#dependencies
import os
from io import StringIO
import numpy as np
import pandas as pd

#for ellipse fitting
from math import atan2
from numpy.linalg import eig, inv, svd

#data smoothing
from scipy import signal

#metpy related dependencies - consider removing entirely
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
from metpy.units import units

#tk gui
import tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter.font import Font

###############################BEGINING OF USER INPUT##########################

#Which functionality would you like to use?
showVisualizations = True     # Displays macroscopic hodograph for flight
siftThruHodo = False    # Use manual GUI to locate ellipse-like structures in hodograph
analyze = True    # Display list of microhodographs with overlayed fit ellipses as well as wave parameters
location = "Tolten"     #[Tolten]/[Villarica]

#variables that are specific to analysis: These might be changed regularly depending on flight location, file format, etc.
flightData = ""             #flight data directory
fileToBeInspected = ''      #specific flight profile to be searched through manually
microHodoDir = ""           #location where selections from GUI are saved. This is also the location where do analysis looks for micro hodos to analysis
waveParamDir = ""             #location where wave parameter files are to be saved

if location == "Tolten":
    latitudeOfAnalysis = abs(-39.236248) * units.degree    #latitude at which sonde was launched. Used to account for affect of coriolis force.
elif location == "Villarica":
    latitudeOfAnalysis = abs(-39.30697) * units.degree     #same, but for Villarica...

g = 9.8                     #m * s^-2
heightSamplingFreq = 5      #used in determining moving ave filter window
minAlt = 1000 * units.m     #minimun altitude of analysis
p_0 = 1000 * units.hPa      #needed for potential temp calculatiion
movingAveWindow = 11        #need to inquire about window size selection

##################################END OF USER INPUT######################

def preprocessDataNoResample(file, path):
    """ prepare data for hodograph analysis. non numeric values & values > 999999 removed, brunt-viasala freq
        calculated, background wind removed
    """
 
    #indicate which file is in progress
    print("Analyzing: {}".format(file))
    
    # Open file
    contents = ""
    f = open(os.path.join(path, file), 'r')
    print("\nOpening file "+file+":")
    for line in f:  # Iterate through file, line by line
        if line.rstrip() == "Profile Data:":
            contents = f.read()  # Read in rest of file, discarding header
            print("File contains GRAWMET profile data")
            break
    f.close()  # Need to close opened file


    # Read in the data and perform cleaning
    # Need to remove space so Virt. Temp reads as one column, not two
    contents = contents.replace("Virt. Temp", "Virt.Temp")
    # Break file apart into separate lines
    contents = contents.split("\n")
    contents.pop(1)  # Remove units so that we can read table
    index = -1  # Used to look for footer
    for i in range(0, len(contents)):  # Iterate through lines
        if contents[i].strip() == "Tropopauses:":
            index = i  # Record start of footer
    if index >= 0:  # Remove footer, if found
        contents = contents[:index]
    contents = "\n".join(contents)  # Reassemble string

    # format flight data in dataframe
    data = pd.read_csv(StringIO(contents), delim_whitespace=True)
    
    #turn strings into numeric data types, non numerics turned to nans
    data = data.apply(pd.to_numeric, errors='coerce') 

    # replace all numbers greater than 999999 with nans
    data = data.where(data < 999999, np.nan)    

    #truncate data at greatest alt
    data = data[0 : np.where(data['Alt']== data['Alt'].max())[0][0]+1]  
    print("Maximum Altitude: {}".format(max(data['Alt'])))

    #drop rows with nans
    data = data.dropna(subset=['Time', 'T', 'Ws', 'Wd', 'Long.', 'Lat.', 'Alt'])

    #change data container name, sounds silly but useful for troubleshooting data-cleaning bugs
    df = data

    #make following vars availabale outside of function - convenient for time being, but consider changing in future
    global Time 
    global Pres 
    global Temp 
    global Hu 
    global Wd 
    global Long 
    global Lat 
    global Alt 
    global Geopot 
    global potentialTemp
    global bv2
    global u, v 
    global uBackground 
    global vBackground
    global tempBackground
    
    #individual series for each variable, local
    Time = df['Time'].to_numpy()
    Pres = df['P'].to_numpy() * units.hPa
    Temp = df['T'].to_numpy()  * units.degC
    Hu = df['Hu'].to_numpy()
    Ws = df['Ws'].to_numpy() * units.m / units.second
    Wd = df['Wd'].to_numpy() * units.degree
    Long = df['Long.'].to_numpy()
    Lat = df['Lat.'].to_numpy()
    Alt = df['Alt'].to_numpy() * units.meter
    Geopot = df['Geopot'].to_numpy()
    

    #calculate brunt-viasala frequency **2 
    tempK = Temp.to('kelvin')
    potentialTemperature =  tempK * (p_0 / Pres) ** (2/7)    #https://glossary.ametsoc.org/wiki/Potential_temperature   
    #bv2 = mpcalc.brunt_vaisala_frequency_squared(Alt, potTemp)    #N^2 
    bv2 = bruntViasalaFreqSquared(potentialTemperature, heightSamplingFreq)     #Maybe consider using metpy version of N^2 ? Height sampling is not used in hodo method, why allow it to affect bv ?

    #convert wind from polar to cartesian c.s.
    u, v = mpcalc.wind_components(Ws, Wd)   #raw u,v components - no different than using trig fuctions
    
    # run moving average over u,v comps
    altExtent = max(Alt) - minAlt    #NEED TO VERIFY THE CORRECT WINDOW SAMPLING SZE
    window = int((altExtent.magnitude / (heightSamplingFreq * 4)))    # as done in Tom's code; arbitrary at best. removed choosing max between calculated window and 11,artifact from IDL code
    if (window % 2) == 0:       #many filters require odd window
        window = window-1
        
    
    #de-trend u, v, temp series; NEED TO RESEARCH MORE, rolling average vs. fft vs. polynomial fit vs. others?
    uBackground = signal.savgol_filter(u.magnitude, window, 3, mode='mirror') * units.m/units.second        #savitsky-golay filter fits polynomial to moving window, has advantage of preserving high frequency signal contnt, compared to 
    vBackground = signal.savgol_filter(v.magnitude, window, 3, mode='mirror') * units.m/units.second
    tempBackground = signal.savgol_filter(Temp.magnitude, window, 3, mode='mirror') * units.degC
    
    #subtract background
    u -= uBackground
    v -= vBackground
    Temp -= tempBackground
    
    return 

def bruntViasalaFreqSquared(potTemp, heightSamplingFreq):
    """ replicated from Tom's script
    """
    G = 9.8 * units.m / units.second**2
    N2 = (G / potTemp) * np.gradient(potTemp, heightSamplingFreq * units.m)     #artifact of tom's code, 
    return N2

class microHodo:
    def __init__(self, ALT, U, V, TEMP, BV2, LAT, LONG, TIME):
      self.alt = ALT#.magnitude
      self.u = U#.magnitude
      self.v = V#.magnitude
      self.temp = TEMP#.magnitude
      self.bv2 = BV2#.magnitude
      self.lat = LAT
      self.long = LONG
      self.time = TIME
      
      
    def addNameAddPath(self, fname, fpath):
        #adds file name attribute to object
        self.fname = fname
        self.savepath = fpath
        
    def addAltitudeCharacteristics(self):
        self.lowerAlt = min(self.alt).astype('int')
        self.upperAlt = max(self.alt).astype('int')
      
    def getParameters(self):

        #Altitude of detection - mean
        self.altOfDetection = np.mean(self.alt)     # (meters)

        #Latitude of Detection - mean
        self.latOfDetection = np.mean(self.lat)     # (decimal degrees) 

        #Longitude of Detection - mean
        self.longOfDetection = np.mean(self.long)     # (decimal degrees)

        #Date/Time of Detection - mean - needs to be added!

        #Axial ratio
        wf = (2 * self.a) / (2 * self.b)    #long axis / short axis

        #Vertical wavelength
        self.lambda_z = self.alt[-1] - self.alt[0]       # (meters) -- Toms script multiplies altitude of envelope by two? 
        self.m = 2 * np.pi / self.lambda_z      # vertical wavenumber (rad/meters)

        #Horizontal wavelength
        bv2Mean = np.mean(self.bv2)
        coriolisFreq = mpcalc.coriolis_parameter(latitudeOfAnalysis)
        
        k_h = np.sqrt((coriolisFreq.magnitude**2 * self.m**2) / abs(bv2Mean) * (wf**2 - 1)) #horizontal wavenumber (1/meter)
        self.lambda_h = 1 / k_h     #horizontal wavelength (meter)

        #Propogation Direction (Marlton 2016) 
        #rot = np.array([[np.cos(self.phi), -np.sin(self.phi)], [np.sin(self.phi), np.cos(self.phi)]])       #2d rotation matrix - containinng angle of fitted elipse - verbatim from Toms script
        rot = np.array([[np.cos(-self.phi), -np.sin(-self.phi)], [np.sin(-self.phi), np.cos(-self.phi)]])       #2d rotation matrix - containinng  negative pose of fitted ellipse, this is opposite from matlab script,
        uv = np.array([self.u, self.v])       #zonal and meridional components                                      since matlab script reports ellipse pose as CW=positive
        uvrot = np.matmul(rot,uv)       #change of coordinates
        urot = uvrot[0,:]               #urot aligns with major axis
        dt = np.diff(self.temp)
        dz = np.diff(self.alt)
        dTdz = np.diff(self.temp)  / np.diff(self.alt)  #discreet temperature gradient dt/dz           
        eta = np.mean(dTdz / urot[0:-1])
        if eta < 0:                 # check to see if temp perterbaton has same sign as u perterbation - clears up 180 deg ambiguity in propogation direction
            self.phi += np.pi
        
        self.directionOfPropogation = self.phi      # (radians ccw fromxaxis)
        self.directionOfPropogation = np.rad2deg(self.directionOfPropogation)
        #self.directionOfPropogation = 450 - self.directionOfPropogation    #use this to convert to degrees est of north -ie compass azmith
        if self.directionOfPropogation > 360:
            self.directionOfPropogation -= 360

        
        
        intrinsicFreq = coriolisFreq.magnitude * wf     #one ought to assign units to output from ellipse fitting to ensure dimensional accuracy
        intrinsicHorizPhaseSpeed = intrinsicFreq / k_h

        #extraneous calculations - part of Tom's script - also list of params that would be nice to add
        #k_h_2 = np.sqrt((intrinsicFreq**2 - coriolisFreq.magnitude**2) * (self.m**2 / abs(bv2Mean)))
        #int2 = intrinsicFreq / k_h_2
        #Intrinsic vertical group velocity
        #Intrinsic horizontal group velocity
        #Intrinsic vertical phase speed
        #Intrinsic horizontal phase speed (m/s)

       
        
        return  [self.altOfDetection, self.latOfDetection, self.longOfDetection, self.lambda_z, k_h, intrinsicHorizPhaseSpeed, wf, self.directionOfPropogation]

    def saveMicroHodoNoIndices(self):
        """ dumps microhodograph object attributs into csv 
        """
    
        T = np.column_stack([self.time, self.alt.magnitude, self.u.magnitude, self.v.magnitude, self.temp.magnitude, self.bv2.magnitude, self.lat, self.long])
        T = pd.DataFrame(T, columns = ['time', 'alt', 'u', 'v', 'temp', 'bv2', 'lat','long'])
        
        fname = '{}_microHodograph_{}-{}'.format(self.fname.strip('.txt'), int(self.alt[0].magnitude), int(self.alt[-1].magnitude))
        T.to_csv('{}/{}.csv'.format(self.savepath, fname), index=False)                          

    #ellipse fitting courtesy of  Nicky van Foreest https://github.com/ndvanforeest/fit_ellipse
    # a least squares algorithm is used
    def ellipse_center(self, a):
        """@brief calculate ellipse centre point
        @param a the result of __fit_ellipse
        """
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        num = b * b - a * c
        x0 = (c * d - b * f) / num
        y0 = (a * f - b * d) / num
        return np.array([x0, y0])
    
    
    def ellipse_axis_length(self, a):
        """@brief calculate ellipse axes lengths
        @param a the result of __fit_ellipse
        """
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) *\
                ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        down2 = (b * b - a * c) *\
                ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        res1 = np.sqrt(up / down1)
        res2 = np.sqrt(up / down2)
        return np.array([res1, res2])
    
    
    def ellipse_angle_of_rotation(self, a):
        """@brief calculate ellipse rotation angle
        @param a the result of __fit_ellipse
        """
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        return atan2(2 * b, (a - c)) / 2
    
    def fmod(self, x, y):
        """@brief floating point modulus
            e.g., fmod(theta, np.pi * 2) would keep an angle in [0, 2pi]
        @param x angle to restrict
        @param y end of  interval [0, y] to restrict to
        """
        r = x
        while(r < 0):
            r = r + y
        while(r > y):
            r = r - y
        return r
    
    
    def __fit_ellipse(self, x,y):
        """@brief fit an ellipse to supplied data points
                    (internal method.. use fit_ellipse below...)
        @param x first coordinate of points to fit (array)
        @param y second coord. of points to fit (array)
        """
        x, y = x[:, np.newaxis], y[:, np.newaxis]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
        S, C = np.dot(D.T, D), np.zeros([6, 6])
        C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
        U, s, V = svd(np.dot(inv(S), C))
        return U[:, 0]
    
    
    def fit_ellipse(self):
        """@brief fit an ellipse to supplied data points: the 5 params
            returned are:
            a - major axis length
            b - minor axis length
            cx - ellipse centre (x coord.)
            cy - ellipse centre (y coord.)
            phi - rotation angle of ellipse bounding box
        @param x first coordinate of points to fit (array)
        @param y second coord. of points to fit (array)
        """
        x, y = self.u, self.v
        e = self.__fit_ellipse(x,y)
        centre, phi = self.ellipse_center(e), self.ellipse_angle_of_rotation(e)
        axes = self.ellipse_axis_length(e)
        a, b = axes
    
        # assert that a is the major axis (otherwise swap and correct angle)
        if(b > a):
            tmp = b
            b = a
            a = tmp
    
        # ensure the angle is betwen 0 and 2*pi
        phi = self.fmod(phi, 2. * np.pi)   #originally alpha = ...
            
        self.a = a
        self.b = b
        self.c_x = centre[0]
        self.c_y = centre[1]
        self.phi = phi
        return 

def doAnalysis(microHodoDir):
    """ Extracts wave parameters from microHodographs; this function can be run on existing microhodograph files without needing to operate the GUI
    """
    #make sure files are retrieved from correct directory; consider adding additional checks to make sure user is querying correct directory
    print("Micro Hodograph Path Exists: ", os.path.exists(microHodoDir))
    
    hodo_list = []
    parameterList = []
    print("all files in path:", os.listdir(microHodoDir))
    for file in os.listdir(microHodoDir):
        path = os.path.join(microHodoDir, file)
        print('Analyzing micro-hodos for flight: {}'.format(microHodoDir))
        
        #dataframe from local hodograph file
        df = np.genfromtxt(fname=path, delimiter=',', names=True)
    
        #create microhodograph object, then start giving it attributes
        instance = microHodo(df['Alt'], df['u'], df['v'], df['temp'], df['bv2'], df['lat'], df['long'], df['time'])

        #file name added to object attribute here to be used in labeling plots
        instance.addNameAddPath(file, microHodoDir)  

        #find out min/max altitudes file
        instance.addAltitudeCharacteristics()
        
        #lets try to fit an ellipse to microhodograph
        instance.fit_ellipse()
        
        #use ellipse to extract wave characteristics
        params = instance.getParameters()
        print("Wave Parameters: \n", params)

        #update running list of processed hodos and corresponding parameters
        parameterList.append(params)
        hodo_list.append(instance)  #add micro-hodo to running list
    
    #organize parameters into dataframe; dump into csv
    parameterList = pd.DataFrame(parameterList, columns = ['Altitude of Detection', 'Lat', 'Long', 'Vert Wavelength', 'Horizontal Wave#', 'IntHorizPhase Speed', 'Axial Ratio L/S', 'Propagation Direction' ])
    parameterList.sort_values(by='Altitude of Detection', inplace=True)
    
    pathAndFile = "{}\{}_params.csv".format(waveParamDir, fileToBeInspected.strip(".txt"))
    parameterList.to_csv(pathAndFile, index=False, na_rep='NaN')
    
    #sort list of hodographs in order of ascending altitude 
    hodo_list.sort(key=lambda x: x.altOfDetection)  

    return hodo_list     
    
def plotBulkMicros(hodo_list, fname):
    """ plot microhodographs in grid of subplots
    """ 
    #plots all micro-hodographs for a single flight
    bulkPlot = plt.figure(fname, figsize=(8.5,11))
    plt.suptitle("Micro-hodographs for \n {}".format(fname))#, y=1.09)
    
    
    totalPlots = len(hodo_list)
    if totalPlots > 0:
        
        #figure out how to arrang subplots on grid
        numColumns = np.ceil(np.sqrt(totalPlots)).astype('int')
        numRows = np.ceil((totalPlots / numColumns)).astype('int')
        position = range(1, totalPlots + 1)
        
        i = 0   #counter for indexing micro-hodo objects
        for hodo in hodo_list:
            #print("HODO ITERATION: ", hodo)
            ax = bulkPlot.add_subplot(numRows, numColumns, position[i], aspect='equal')
            ax.plot(hodo_list[i].u, hodo_list[i].v) 
        
            #plot parametric best fit ellipse
            param = np.linspace(0, 2 * np.pi)
            x = hodo_list[i].a * np.cos(param) * np.cos(hodo_list[i].phi) - hodo_list[i].b * np.sin(param) * np.sin(hodo_list[i].phi) + hodo_list[i].c_x
            y = hodo_list[i].a * np.cos(param) * np.sin(hodo_list[i].phi) + hodo_list[i].b * np.sin(param) * np.cos(hodo_list[i].phi) + hodo_list[i].c_y
            ax.plot(x, y)
            ax.set_xlabel("(m/s)")
            ax.set_ylabel("(m/s)")
            ax.set_aspect('equal')
            ax.set_title("{}-{} (m)".format(hodo_list[i].lowerAlt, hodo_list[i].upperAlt), fontsize=14 )
            i += 1
        
        plt.subplots_adjust(top=.9, hspace=.5)            
        plt.show() 
        return

def macroHodo():
    """ plot hodograph for entire flight
    """
    #plot v vs. u
    plt.figure("Macroscopic Hodograph", figsize=(10,10))  #Plot macroscopic hodograph
    #c=Alt
    #plt.scatter( u, v, c=c, cmap = 'magma', s = 1, edgecolors=None, alpha=1)
    plt.plot(u,v)
    #cbar = plt.colorbar()
    #cbar.set_label('Altitude')  
    return

'''Interesting book recomendation: The tipping point by Malcolm Gladwell // Michael Pollan [coffee?]; per Dave Brown's reccommendation...'''

def uvVisualize():
    """ show u, v, background wind vs. altitude
    """
    #housekeeping
    plt.figure("U & V vs Time", figsize=(10,10)) 
    plt.suptitle('Smoothed U & V Components', fontsize=16)

    #u vs alt
    plt.subplot(1,2,1)
    plt.plot((u.magnitude + uBackground.magnitude), Alt.magnitude, label='Raw')
    plt.plot(uBackground.magnitude, Alt.magnitude, label='Background')
    plt.plot(u.magnitude, Alt.magnitude, label='De-Trended')
    plt.xlabel('(m/s)', fontsize=12)
    plt.ylabel('(m)', fontsize=12)
    plt.title("U")

    #v vs alt
    plt.subplot(1,2,2)
    plt.plot((v.magnitude + vBackground.magnitude), Alt.magnitude, label='Raw')
    plt.plot(vBackground.magnitude, Alt.magnitude, label='Background')
    plt.plot(v.magnitude, Alt.magnitude, label='De-Trended')
    plt.xlabel('(m/s)', fontsize=12)
    plt.ylabel('(m)', fontsize=12)
    plt.legend(loc='upper right', fontsize=16)
    plt.title("V")
    return



def manualTKGUI():
    """ tool for visualizing hodograph, sifting for micro hodographs, saving files for future analysis
    """
    
    class App:
        def __init__(self, master):
            """ method sets up gui
            """
            
            alt0 = 0
            wind0 = 100
            # Create a container
            tkinter.Frame(master)
            
            #Create Sliders
            self.alt = IntVar()
            self.win = IntVar()

            #initialize gui vars added 12/27
            #self.alt.set(min(Alt.magnitude.tolist())) 
            
            #self.altSpinner = tkinter.Spinbox(root, command=self.update, textvariable=self.alt, values=Alt.magnitude.tolist(), font=Font(family='Helvetica', size=25, weight='normal')).place(relx=.05, rely=.12, relheight=.05, relwidth=.15)
            #added 12/27 test
            self.altSpinner = tkinter.Spinbox(root, command=self.update, values=Alt.magnitude.tolist(), font=Font(family='Helvetica', size=25, weight='normal'))
            self.altSpinner.place(relx=.05, rely=.12, relheight=.05, relwidth=.15)  #originally followed above line
            #self.winSpinner = tkinter.Spinbox(root, command=self.update, textvariable=self.win, from_=5, to=1000, font=Font(family='Helvetica', size=25, weight='normal')).place(relx=.05, rely=.22, relheight=.05, relwidth=.15)
            self.winSpinner = tkinter.Spinbox(root, command=self.update, from_=5, to=1000, font=Font(family='Helvetica', size=25, weight='normal'))
            self.winSpinner.place(relx=.05, rely=.22, relheight=.05, relwidth=.15)  #originally followed above line
            self.altLabel = tkinter.Label(root, text="Select Lower Altitude (m):", font=Font(family='Helvetica', size=18, weight='normal')).place(relx=.05, rely=.09)
            self.winLabel = tkinter.Label(root, text="Select Alt. Window (# data points):", font=Font(family='Helvetica', size=18, weight='normal')).place(relx=.05, rely=.19)
            
            #Create figure, plot 
            fig = Figure(figsize=(5, 4), dpi=100)
            self.ax = fig.add_subplot(111)
            fig.suptitle("{}".format(fileToBeInspected))
            self.l, = self.ax.plot(u[:alt0+wind0], v[:alt0+wind0], 'o', ls='-', markevery=[0])
            self.ax.set_aspect('equal')
        
            self.canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
            self.canvas.draw()
            self.canvas.get_tk_widget().place(relx=0.25, rely=0.1, relheight=.8, relwidth=.6)
            #frame.pack()
            
            self.winLabel = tkinter.Label(root, text="Blue dot indicates lower altitude", font=Font(family='Helvetica', size=15, weight='normal')).place(relx=.05, rely=.3)
            self.quitButton = tkinter.Button(master=root, text="Quit", command=self._quit).place(relx=.05, rely=.6, relheight=.05, relwidth=.15)
            self.saveButton = tkinter.Button(master=root, text="Save Micro-Hodograph", command=self.save).place(relx=.05, rely=.5, relheight=.05, relwidth=.15)

            self.readyToSave = False #flag to make sure hodo is updated before saving
            #---------
            
        def update(self, *args):
            """ on each change to gui, this method refreshes hodograph plot
            """
            self.readyToSave = True
            #sliderAlt = int(self.alt.get()) works originally
            sliderAlt = int(float(self.altSpinner.get()))
            sliderWindow = int(self.winSpinner.get())
            self.l.set_xdata(u[np.where(Alt.magnitude == sliderAlt)[0][0]:np.where(Alt.magnitude == sliderAlt)[0][0] + sliderWindow])
            self.l.set_ydata(v[np.where(Alt.magnitude == sliderAlt)[0][0]:np.where(Alt.magnitude == sliderAlt)[0][0] + sliderWindow])
           
            self.ax.autoscale(enable=True)
            #self.ax.autoscale
            self.ax.relim()
            self.canvas.draw()
            return
        
        def save(self): 
            """ save current visible hodograph to .csv for further analysis
            """
            if self.readyToSave:

                sliderAlt = int(float(self.altSpinner.get()))
                sliderWindow = int(float(self.winSpinner.get()))
                lowerAltInd = np.where(Alt.magnitude == sliderAlt)[0][0]
                upperAltInd = lowerAltInd + sliderWindow
            
                #collect local data for altitude that is visible in plot, dump into .csv
                ALT = Alt[lowerAltInd : upperAltInd]
                U = u[lowerAltInd : upperAltInd]
                V = v[lowerAltInd : upperAltInd]
                TEMP = Temp[lowerAltInd : upperAltInd]
                BV2 = bv2[lowerAltInd : upperAltInd]
                LAT = Lat[lowerAltInd : upperAltInd]
                LONG = Long[lowerAltInd : upperAltInd]
                TIME = Time[lowerAltInd : upperAltInd]
                instance = microHodo(ALT, U, V, TEMP, BV2, LAT, LONG, TIME)
                instance.addNameAddPath(fileToBeInspected, microHodoDir)
                instance.saveMicroHodoNoIndices()
                
            else: 
                print("Please Update Hodograph")
                    
            
            return

        def _quit(self):
            """ terminate gui
            """
            root.quit()     # stops mainloop
            root.destroy()  # this is necessary on Windows to prevent Fatal Python Error: PyEval_RestoreThread: NULL tstate
    
    root = tkinter.Tk()
    root.geometry("400x650")
    root.wm_title("Manual Hodograph GUI")
    App(root)
    root.mainloop()
    return

def run_(file, filePath):
    """ call neccesary functions 
    """
    #make sure there are no existing figures open
    plt.close('all')

    # set location of flight data as surrent working directory
    os.chdir(filePath)
    preprocessDataNoResample(file, flightData)
    
    if showVisualizations:
        macroHodo()
        uvVisualize()
        
    if siftThruHodo:
       manualTKGUI()
       
    if analyze:
        hodo_list= doAnalysis(microHodoDir)
        plotBulkMicros(hodo_list, file)
        
    return
     
#Calls run_ method  
run_(fileToBeInspected, flightData) 
    
#last line