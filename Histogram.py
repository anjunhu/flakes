from __future__ import print_function
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os
import cv2
import heapq
import argparse
from scipy.signal import argrelextrema, find_peaks, find_peaks_cwt
from operator import itemgetter
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import collections
import time
from scipy import eye
import scipy.interpolate
#from matplotlib import pylab
#from matplotlib.ticker import FormatStrFormatter

class Histogram:
    def __init__(self, fn_original, src_path, out_path, stats_out_path=None, obj_mag=50):
        self.obj_mag = obj_mag
        self.fn_original = fn_original
        self.path_original = src_path+'\\'+fn_original
        #self.histogram = None
        #self.smoothedHistogram = self.drawSmoothedHistogram()
        #self.out_path_3 = out_path+'\\3_Global_Histogram'
        self.out_path_6 = out_path+'\\4_Local_Histogram'
        self.out_path_local = out_path
        #self.stats_out_path = stats_out_path
        #self.saveLocalHistogram()
        
    def drawHistogramRGB (self):
        img = cv2.imread(self.path_original)
        color = ('blue','green','red')
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,260])
        return histr
 

    def saveLocalHistogram (self,ftr, material='C',smooth=False, presentation=2):
        
        img = cv2.imread(self.path_original)
        img = cv2.medianBlur(img,5)
        intensity = None
        filterstr = 'gray'
        b, g, r = cv2.split(img)
        
        if ftr==0:      #ND
            intensity = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        elif ftr<500:   #B
            filterstr = 'blue'
        elif ftr>600:   #R
            intensity = r
            filterstr = 'red'
            if np.percentile(r.flatten(),90) > 160:
                return '',''
        else:           #G
            intensity = g
            filterstr = 'green'
        #intensity = cv2.medianBlur(intensity,5)  
        w, h = intensity.shape[::-1]
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        fig = plt.figure(1)
        plt.subplot2grid((2,3), (0,0))
        plt.xticks(())
        plt.yticks(())
        plt.title(filterstr)
        plt.imshow(intensity, cmap = 'gray')
        
        plt.subplot2grid((2,3), (0,2))
        plt.xticks(())
        plt.yticks(())
        plt.title(self.fn_original)
        plt.imshow(img)
        
        edgemask = np.zeros_like(intensity)
        laplacian = cv2.Laplacian(intensity,cv2.CV_64F)
        edgemask[laplacian <= 3 ] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        edgemask = cv2.erode(edgemask,kernel,iterations = 1)
        #cv2.imshow('edgemask', edgemask)
        #cv2.waitKey(0)
        

    
    
        intensity[edgemask==0] = 255
        plt.subplot2grid((2,3), (0,1))
        plt.xticks(())
        plt.yticks(())
        plt.title('edgemask')
        plt.imshow(intensity, cmap = 'gray')
        #plt.show()
        
        
        
        histr = cv2.calcHist([intensity],[0],edgemask,[256],[0,256])
        histr = histr.ravel()
        histr_smoothed = histr

        
        if smooth:
            #histr_smoothed = self.smooth(histr)
            histr_smoothed = savgol_filter(histr, 5, 3)
            #histr_smoothed = lowess(histr, range(0,256), frac=0.1,return_sorted=False)
            
        #print (str(histr))
        if smooth:
            prom = (w+h)/10
        else:
            prom = (w+h)/4
        noisypeaks, properties = find_peaks(histr_smoothed, prominence = 1, height = 1, distance = 3)
        proms = properties["prominences"]
        hghts = properties["peak_heights"]
        prominence_threshold = min(np.percentile(proms,75),max(proms)/25)
        height_threshold = min(np.percentile(hghts,50),max(hghts)/25)
        peaks = []
        peaks_with_prom = {}
        for pk, pm, ht in zip(noisypeaks, proms, hghts):
            if pm > prominence_threshold and ht > height_threshold:
                peaks_with_prom.update({pk:int(pm)} )
                #peaks_with_prom.update({pk:int(pm*ht)} )
                
        peaks_with_prom = sorted(peaks_with_prom.items(), key=itemgetter(1),reverse=True)
        for pwp in peaks_with_prom:
            if len(peaks)<3:
                peaks.append(pwp[0])
            
        #print(peaks)
        #peaks = find_peaks_cwt(histr_smoothed, np.arange(1,3))
        substrate = np.argmax(histr_smoothed)
        contrast_list = []
        for peak in peaks:
            if peak != substrate:
                contrast = (peak-substrate)/substrate
                if contrast < 0.5 or ftr<600:
                    contrast_list.append(round(contrast, 4))
        if len(contrast_list)>0  and material=='C' and min(contrast_list)>1.9:
            pass#return None
        plt.subplot2grid((2,3), (1,0), colspan=3)
        plt.xticks(peaks)
        if smooth:
            plt.plot(histr, dashes=[3, 1],color='gray')
        plt.plot(histr_smoothed,color=filterstr)
        plt.xlim([0,256])
        #plt.xlim([min(peaks)/1.5,max(peaks)*1.5])
        plt.xlabel('Intensity')
        plt.ylabel('Number of Pixels')
        layerint_list = []
        if ftr>0:
            layer_list = []
            max_layers = 16
            if ftr<600:
                max_layers = 41
            
            for i in range(0,max_layers):
                layer_list.append(self.multilayer_contrast(numlayers=i,wavelengthnm=ftr))
                
            interp = scipy.interpolate.interp1d(layer_list, range(0,max_layers))
            for contrast in contrast_list:
                if (contrast > min(layer_list) and contrast<max(layer_list)):
                    layerfloat = interp(contrast)
                    layerint_list.append(int(layerfloat+1))
            layerint_list = list(set(layerint_list))
            plt.title('Contrast='+str(contrast_list)+' >> EstLayers='+str(layerint_list))
        else:
            plt.title('Contrast='+str(contrast_list))
        plt.locator_params(axis='y', nbins=5)
        
        
        fig.tight_layout()
        
        if presentation:
            self.move_figure(fig,50,200)
            plt.show(block=False)
            plt.pause(presentation)
            plt.close()
        

            

        #fig.set_size_inches(w=11,h=7)
        if not os.path.exists(self.out_path_local):
            os.makedirs(self.out_path_local)
        fig.savefig(os.path.join(self.out_path_local , (self.fn_original.split('.')[0]+'.png')))
        
        contrast_list_str = str(contrast_list).replace(",", "|")
        layerint_list_str = str(layerint_list).replace(",", "|")
        
        
        return contrast_list_str, layerint_list_str
            
    
    def multilayer_contrast(self, numlayers=1, SiO2thicknessnm=300, wavelengthnm=542):
        if wavelengthnm==542:
            n1 = 2.6809-1.235*1j #graphite
            n2 = 1.4794 #SiO2
            n3 = 4.1 - 1j*0.03 #Si
        elif wavelengthnm==610:
            n1 = 2.7131-1j*1.3256
            n2 = 1.4768 #SiO2
            n3 = 3.908 - 1j*0.017257 #Si
        I_substrate = self.Intensity(1,n2,n3,numlayers,SiO2thicknessnm,wavelengthnm)  
        I_graphite = self.Intensity(n1,n2,n3,numlayers,SiO2thicknessnm,wavelengthnm) 
        return (I_graphite - I_substrate) / I_substrate
        
            
    def Intensity (self,n1,n2,n3,numlayers=1,SiO2thicknessnm=300,wavelengthnm=542):
        n0 = 1
        r1 = (n0-n1)/(n0+n1)
        r2 = (n1-n2)/(n1+n2)
        r3 = (n2-n3)/(n2+n3)
        d2 = SiO2thicknessnm*1e-9
        d1 = 0.335e-9*numlayers
        wavelength = 1e-9*wavelengthnm
        Phi1 = 2*np.pi*n1*d1/wavelength
        Phi2 = 2*np.pi*n2*d2/wavelength
        Gamma = (r1*np.exp(1j*(Phi1+Phi2))+r2*np.exp(-1j*(Phi1-Phi2))+\
                 +r3*np.exp(-1j*(Phi1+Phi2))+ r1*r2*r3*np.exp(1j*(Phi1-Phi2)))*\
                 (np.exp(1j*(Phi1+Phi2))+r1*r2*np.exp(-1j*(Phi1-Phi2))+\
                  r1*r3*np.exp(-1j*(Phi1+Phi2))+r2*r3*np.exp(1j*(Phi1-Phi2)))**-1
        return (np.abs(Gamma))**2
    
    
    def sumAbove (self, histogram):
        smtotal = 0
        smabv = 0
        #line_space = np.linspace(0, 256, num=257, endpoint=True)
        hist_abv = histogram[150:]
        for weight in histogram:
            smtotal += weight
        #print (smtotal)
        #line_space = np.linspace(170, 256, num=87, endpoint=True)
        for weight in hist_abv:
            smabv += weight
        #print (sm170)
        return smabv/smtotal, smtotal
    
    #https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    def smooth(self,x,window_len=5,window='hamming'):
        if x.ndim != 1:
            raise (ValueError, "smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
            raise (ValueError, "Input vector needs to be bigger than window size.")
        if window_len<3:
            return x   
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='valid')
        return y
    
    def move_figure(self, f, x, y):
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        elif backend == 'WXAgg':
            f.canvas.manager.window.SetPosition((x, y))
        else:
            f.canvas.manager.window.move(x, y)
        
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('fn_original',type=str)
        parser.add_argument('src_path',type=str)
        parser.add_argument('out_path',type=str)
        
        args = parser.parse_args()
        
        fn_original = args.fn_original
        src_path = args.src_path
        out_path = args.out_path
    except:
        pass
    Histogram(fn_original, src_path, out_path)