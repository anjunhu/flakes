'''
!!! Camera-Stage-Sample coordinates should align !!!
!!! Make sure the gears are not rubbing anything (wires/microscope/etc.) !!!

Please follow the instructions in the GUI and restore axis position if necessary

Command parsing funtions for picocontroller driver are written by ** Ben Hammel ** 
https://github.com/bdhammel

.NET/C++/C drivers and their Python wrappers are provided by the manufacturer 
of the Nikon Eclipse FN1 Microscope camera, ** The Imaging Source, LLC **


Current Placement
x+ = 1+
y+ = 2-

'''


from __future__ import print_function
import os
import cv2
import numpy as np
import math
import re
import time
import usb.core
import usb.util
import ctypes as C
import tisgrabber as IC
import random
from operator import itemgetter
import Histogram, Contour, Rectangle
import statistics
    
class BayerAcq:
    
    def __init__(self, out_path, minX=4, minY=3, maxX=4, maxY=3, ftr=542, objective_magnification=50, material='C',presentation=0):
        # file I/O
        self.presentation = presentation
        self.material = material
        self.filter = ftr
        self.result_path = out_path
        self.out_path = out_path+'\\1_Image_Acquisition\\'
        self.out_path_bayer = out_path+'\\1_Image_Acquisition_Bayer\\'
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        if not os.path.exists(self.out_path_bayer):
            os.makedirs(self.out_path_bayer)
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY
        self.maxx = 5
        self.maxy = 5
        self.posX = minX
        self.posY = minY
        self.posx = 1
        self.posy = 1
        self.moved_in_X_not_Y = True
        # picocontroller
        self.NEWFOCUS_COMMAND_REGEX = re.compile("([0-9]{0,1})([a-zA-Z?]{2,})([0-9+-]*)")
        self.dev = None
        self.idProduct = 0x4000
        self.idVendor = 0x104d
        self.ep_in = None
        self.ep_out = None
        while self.dev == None:
            try:
                self.GrabPicocontroller()
                self.command('st')
                self.set_home_position()
                self.command('1va2000')
                self.command('2va2000')
                self.command('3va2000')
            except:
                C.windll.user32.MessageBoxW(0, "Please plug in and turn on the picomotor controller!.", "Where are the motors?", 0)
            
        # vulnerable and objective dependent parameters!!  
        self.autofocus_reference = int(200/objective_magnification)
        self.obj_mag = objective_magnification
        if (objective_magnification == 10):
            self.unitsq = 512
            self.MIN_LITHO_X, self.MAX_LITHO_X, self.MIN_LITHO_Y, self.MAX_LITHO_Y = 70, 410, 70, 410
            litholistx = [70, 240, 410]
            litholisty = [70, 240, 410]
            self.LITHO_MATRIX = [(a,b) for a in litholistx for b in litholisty]
            self.LITHO_MATRIX = np.reshape(self.LITHO_MATRIX , (3, 3, 2))
            self.X_NUMBER_POS = [698+50,344+35]
            self.Y_NUMBER_POS = [698+100,344+35]
        elif (objective_magnification == 50):
            self.unitsq = 1200
            self.MIN_LITHO_X, self.MAX_LITHO_X, self.MIN_LITHO_Y, self.MAX_LITHO_Y = 100, 960, 170, 1030
            litholistx = [self.MIN_LITHO_X, self.MAX_LITHO_X]
            litholisty = [self.MIN_LITHO_Y, self.MAX_LITHO_Y]
            self.LITHO_MATRIX = [(a,b) for a in litholistx for b in litholisty]
            self.LITHO_MATRIX = np.reshape(self.LITHO_MATRIX , (2, 2, 2))
            self.X_NUMBER_POS = [320,10]
            self.Y_NUMBER_POS = [320+230,10]
        
        
        LithoExamplarDir = os.path.join( 'LithoExamplar',str(self.obj_mag))
        self.template = cv2.imread(os.path.join(LithoExamplarDir,'cross.png'),0)
        self.coords = []
        self.coords.append(self.template)
        for i in range(1,10):
            temp = cv2.imread(os.path.join(LithoExamplarDir,(str(i)+'.png')),0)
            self.coords.append(temp)
            
        # orthogonalizer matrix: indexed as [majormovingaxis][dir][step_in_y, step_in_x]
        forwardx = [0,5500]
        backwardx = [0,-5200]
        forwardy = [4800,0]
        self.step_mat = [[None,forwardy],[backwardx,forwardx]]
        
        # enable if using on its own
        self.Prepare_and_Acquire()



    def Prepare_and_Acquire (self):        
        ready = False
        while not ready:
            try:
                ready = self.FindMinNumbers()
            except:
                C.windll.user32.MessageBoxW(0, "Please plug in the microscope camera.", "Camera Warning", 0)
        self.command('st')
        self.command('1ac50000')
        self.command('2ac50000')
        self.command('3ac50000')
        try:
            self.Camera = IC.TIS_CAM()
            self.Camera.openVideoCaptureDevice("77020507")
            video_fmt = "Y800 ("+str(self.unitsq)+'x'+str(self.unitsq)+')'
            video_fmt = "Y800 (1920x1200)"
            self.Camera.SetVideoFormat(video_fmt)   
            self.Camera.SetFrameRate( 1.0 )
            self.Camera.SetFormat(IC.SinkFormats.Y800)
            self.Camera.StartLive(1)
            self.step_x_or_y (initial_check=True)
            while (self.posX <= self.maxX and self.posX > 0 and self.posY <= self.maxY):
                self.GetFlakes()
            self.Camera.StopLive()
            print ('Bye!')
            return
        except KeyboardInterrupt:   # ctrl+C to abort in case of disasters
            self.Camera.StopLive()
            self.command('st')
            return
        
        
        

    # (10x) 170 pix = (50x) 860 pixels = 5600 steps
    def FindMinNumbers (self):
        self.command('1ac90000')
        self.command('2ac90000')
        self.command('3ac90000')
        
        self.Camera = IC.TIS_CAM()
        self.Camera.openVideoCaptureDevice("77020507")
        video_fmt = "Y800 (1920x1200)"
        self.Camera.SetVideoFormat(video_fmt)   
        self.Camera.SetFrameRate( 1.0 )
        self.Camera.SetFormat(IC.SinkFormats.Y800)
        self.Camera.StartLive(1)
            
            
        
        
        brightness_checker = cv2.cvtColor(self.crop_image(), cv2.COLOR_RGB2GRAY) #RGBhere!!
        mode = statistics.mode(brightness_checker.flatten())
        while mode<2:
            C.windll.user32.MessageBoxW(0, " Please pull the BINO-PHOTO slider out.", "Bino-photo", 0)
        
            
        C.windll.user32.MessageBoxW(0, " If you cannot see your minX and minY, find them and make sure they are visible from the screen!", "Starting point", 0)
        
        if self.material=='Sb' and self.filter==0:
            mode -= 86
        if self.material =='C' and self.filter==542:
            mode -= 68
        if self.material =='C' and self.filter==610:
            mode -= 76
        if self.material =='C' and self.filter==0:
            mode -= 91
            
        C.windll.user32.MessageBoxW(0, "Try your best to focus on this area!", "Focus", 0)
        
        while abs(mode)>2:
            if mode<-2:
                C.windll.user32.MessageBoxW(0, "Please turn up the lamp (CW) or slide OUT an ND filter.", "It's too dark!", 0)
            else:
                C.windll.user32.MessageBoxW(0, "Please turn down the lamp (CCW) or slide IN an ND filter.", "It's too bright!", 0)
            brightness_checker = cv2.cvtColor(self.crop_image(save=True), cv2.COLOR_BGR2GRAY) 
            mode = statistics.mode(brightness_checker.flatten())
            if self.material=='Sb' and self.filter==0:
                mode -= 86
            if self.material =='C' and self.filter==542:
                mode -= 68
            if self.material =='C' and self.filter==610:
                mode -= 76
            if self.material =='C' and self.filter==0:
                mode -= 91
        
        
        
        topleft_minX, topleft_minY = self.FindNumberMatches()
        while topleft_minX is None and topleft_minY is None:
            topleft_minX, topleft_minY = self.FindNumberMatches()
            
        step_per_pixel_factor = 5000/(self.obj_mag*17)
        # offset in steps
        if topleft_minX is None:
            offsetX = ((topleft_minY[1]-self.Y_NUMBER_POS[1]))*step_per_pixel_factor
            offsetY = int((topleft_minY[0]-self.Y_NUMBER_POS[0])*step_per_pixel_factor)
        elif topleft_minY is None:
            offsetX = ((topleft_minX[1]-self.X_NUMBER_POS[1]))*step_per_pixel_factor
            offsetY = int((topleft_minX[0]-self.X_NUMBER_POS[0])*step_per_pixel_factor)
        else:
            offsetX = ((topleft_minX[1]-self.X_NUMBER_POS[1])+(topleft_minY[1]-self.Y_NUMBER_POS[1]))/2*step_per_pixel_factor
            offsetY = int(((topleft_minX[0]-self.X_NUMBER_POS[0])+(topleft_minY[0]-self.Y_NUMBER_POS[0]))/2)*step_per_pixel_factor


        step_per_pixel_factor = 5000/(self.obj_mag*17)
        self.command('1pr'+str(offsetX))
        time.sleep(max(0.2,abs(offsetX/1200)))
        self.command('2pr'+str(offsetY))
        time.sleep(max(0.2,abs(offsetY/1200)))
        self.command('st')
        
        
        
        
        # check if match is valid
        '''
        topleft_minX, topleft_minY = self.FindNumberMatches()
        while topleft_minX is None and topleft_minY is None:
            topleft_minX, topleft_minY = self.FindNumberMatches()

        # offset in pixels
        
        offsetX = ((topleft_minX[1]-self.X_NUMBER_POS[1])+(topleft_minY[1]-self.Y_NUMBER_POS[1]))/2
        offsetY = int(((topleft_minX[0]-self.X_NUMBER_POS[0])+(topleft_minY[0]-self.Y_NUMBER_POS[0]))/2)
        print ('offsetX = '+str(offsetX)+'    offsetY = '+str(offsetY))
        
        if (offsetX)>(self.unitsq/4) or (offsetY)>(self.unitsq/4):
            C.windll.user32.MessageBoxW(0, "Sorry, a few things could have gone wrong.\nPlease re-check the motors, make sure the field is focused, re-orient the sample and keep it orthogonal.", "Warning", 0)
            self.Camera.StopLive()
            self.Camera = None
            return False
        '''
        self.Camera.StopLive()
        self.Camera = None
        return True


    def FindNumberMatches (self):
        gray_img = self.crop_image(FindNumberMatches=True)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        
        threshold = 0.7
        res = cv2.matchTemplate(gray_img,self.coords[self.minX],cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        matches_minX = []
        for pt in zip(*loc[::-1]):
            distinct_pt = True
            for match in matches_minX:
                if abs(pt[0]-match[0])<(self.unitsq/10) and abs(pt[1]-match[1])<(self.unitsq/10):
                    distinct_pt = False
                    break
            if (distinct_pt):
                matches_minX.append(pt) 
        matches_minX = sorted(matches_minX, key=itemgetter(1))
        
        
        res = cv2.matchTemplate(gray_img,self.coords[self.minY],cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        matches_minY = []
        for pt in zip(*loc[::-1]):
            distinct_pt = True
            for match in matches_minY:
                if abs(pt[0]-match[0])<(self.unitsq/10) and abs(pt[1]-match[1])<(self.unitsq/10):
                    distinct_pt = False
                    break
            '''
            for match in matches_minX:
                if abs(pt[0]-match[0])<(self.unitsq/16) and abs(pt[1]-match[1])<(self.unitsq/16):
                    distinct_pt = False
                    break
            '''
            if (distinct_pt):
                matches_minY.append(pt) 
        matches_minY = sorted(matches_minY, key=itemgetter(1))
        
        if len(matches_minX)<1 and len(matches_minY)<1:
            C.windll.user32.MessageBoxW(0, "I can't see the numbers that you have specified as minX and minY.\nPlease pull the BINO-PHOTO slider out, or reposition the sample", "BINO to PHOTO", 0)
            return None, None
            
        elif (len(matches_minX)<1): 
            return None, matches_minY[0]
        
        elif (len(matches_minY)<1):
            return matches_minX[0], None
        
        return matches_minX[0], matches_minY[0]
            
    # take a picture and then continue touring
    def GetFlakes (self):      
        laplacian_ratio = 1
        # coarseness is determined by how bad the field looks in comparison to the best one so far
        if (self.autofocus_reference > 0):
            self.Camera.SnapImage()
            cropped_gray_img = self.crop_image()
            current_varlaplace = cv2.Laplacian(cropped_gray_img, cv2.CV_64F).var()
            laplacian_ratio = max(1,math.floor(self.autofocus_reference/current_varlaplace))
            '''
        if (laplacian_ratio < 0.5):
            self.command('1pa0')
            self.command('2pa0')
            self.command('3pa0')
            if self.moved_in_X_not_Y:         
                self.posX -= 1
            else:
                self.posY -= 1
            return
            '''
        while (laplacian_ratio >= 1):
            self.AutoFocus_Variance(coarseness=laplacian_ratio)
            laplacian_ratio -= 1
        self.TranslationXY()
        return 

 
    def GrabPicocontroller(self):
        # find the device
        self.dev = usb.core.find(idProduct=self.idProduct, idVendor=self.idVendor)

        if self.dev is None:
            raise ValueError('Device not found')

        # set the active configuration. 
        self.dev.set_configuration()

        # get an endpoint instance
        cfg = self.dev.get_active_configuration()
        intf = cfg[(0,0)]

        self.ep_out = usb.util.find_descriptor(
            intf,
            custom_match = \
            lambda e: \
                usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)
        
        self.ep_in = usb.util.find_descriptor(
            intf,
            custom_match = \
            lambda e: \
                usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

        assert (self.ep_out and self.ep_in) is not None



    def command(self, newfocus_command):
        usb_command = self.parse_command(newfocus_command)
        print ('usb_command = '+usb_command)
        self.ep_out.write(usb_command)



    def parse_command(self, newfocus_command):
        m = self.NEWFOCUS_COMMAND_REGEX.match(newfocus_command)
        # Check to see if a regex match was found in the user submitted command
        if m:
            # Extract matched components of the command
            driver_number, command, parameter = m.groups()
            usb_command = command

            # Construct USB safe command
            if driver_number:
                usb_command = '1>{driver_number} {command}'.format(
                        driver_number=driver_number,
                        command=usb_command)

            if parameter:
                usb_command = '{command} {parameter}'.format(
                        command=usb_command,
                        parameter=parameter)
            usb_command += '\r'
            
            return usb_command
        else:
            print("ERROR! Command {} was not a valid format".format(newfocus_command))    

    # 100 microns = 5500 steps
    def TranslationXY (self):
        print ('Start Translation')
        self.command('st')
        if (self.posY<=self.maxY and self.posy<=self.maxy):
            if ((self.posY+self.posy-self.minY)%2 == 1 and not (self.posX==self.maxX and self.posx==self.maxx) ):
                self.step_x_or_y()
                
            elif ((self.posY+self.posy-self.minY)%2 == 0 and not (self.posX==self.minX and self.posx==1) ):
                self.step_x_or_y(direction=0)
                
            else:
                self.step_x_or_y(moving_axis=0)
                
                
        else:
            return
        
        return
    
    # Placement of stages, as of 2019-06-14
    # 1 mv + = going down = +x
    # 2 mv - = going right = +y
    def step_x_or_y (self, moving_axis=1, direction=1, debug=False, initial_check=False):
        self.command('st')
        if not initial_check:
            # reminder: self.step_mat = [[None,forwardy],[backwardx,forwardx]]
            lastflakename = 'X'+str(self.posX)+'_'+str(self.posx)+'_'+'Y'+str(self.posY)+'_'+str(self.posy) + '.png'
            naptimex = max(0.2,abs((self.step_mat[moving_axis][direction][1])/1200))
            naptimey = max(0.2,abs((self.step_mat[moving_axis][direction][1])/1200))
            
            cmd = '1pr'+str(self.step_mat[moving_axis][direction][1])
            self.command(cmd)
            if (naptimex>naptimey):
                self.analyze(lastflakename)
            else:
                time.sleep(naptimex)
                
            cmd = '2pr'+str(self.step_mat[moving_axis][direction][0])
            self.command(cmd)
            if (naptimey>naptimex):
                self.analyze(lastflakename)
            else:
                time.sleep(naptimey)
        
        if moving_axis == 0:
            time.sleep(2)
            
        current_litho_matrix = self.get_matches()

        offset = [0,0]
        lithocount = 0
        
        while np.count_nonzero(current_litho_matrix) < 1:
            current_litho_matrix = self.get_matches()

        print(str(current_litho_matrix))
        
        
        
        
        for row in range(len(current_litho_matrix)):
            for col in range(len(current_litho_matrix)):
                if current_litho_matrix[row][col][0] !=0 and current_litho_matrix[row][col][1] != 0 :
                    lithocount += 1
                    offset[0] += current_litho_matrix[row][col][0] - self.LITHO_MATRIX[row][col][0] 
                    offset[1] += current_litho_matrix[row][col][1] - self.LITHO_MATRIX[row][col][1] 

        offset[0] /= lithocount # offset in y: if positive, go left (in -y) = 2mv+
        offset[1] /= lithocount # offset in x: if positive, go up (in -x) = 1mv-
        
        offsets = [ [0,0] , offset, offset ]
        print (str(offset))
        
        step_per_pixel_factor = 5000/(self.obj_mag*17)
        offsetX = step_per_pixel_factor*offset[1]
        offsetY = step_per_pixel_factor*offset[0]
        self.command('1pr'+str(offsetX))
        time.sleep(max(0.2,abs(offsetX/1200)))
        self.command('2pr'+str(offsetY))
        time.sleep(max(0.2,abs(offsetY/1200)))
        self.command('st')
        
        # TODO: manual ajustment on the fly
             
        for ofst in offset:
            debug = debug and  ofst>(self.obj_mag*6)
                
        if debug:
            C.windll.user32.MessageBoxW(0, "Adjust with ADWS if needed. \nPress space when you feel ready.", "I have a bad feeling...", 0)
            self.User_Preparation()
            
            
        
        # 5500/170 ~ 32 steps per pixel
        # if offset in x is positive and we are going in +x, we should go further, more positive
        self.step_mat[moving_axis][direction][1] += (offset[1])*4
        self.step_mat[moving_axis][direction][0] += (offset[0])*4
        print ('step_mat update --- '+str(self.step_mat))
        
        step_x = 100
        step_y = 100
        
        
        
        while (abs(offsets[2][0]) > int(2*self.obj_mag/10) or abs(offsets[2][1]) > int(2*self.obj_mag/10)) and (abs(step_x)>1 or abs(step_y)>1):
            print('WHILE...')
            # normalize 
            diag = (offset[1]**2+offset[0]**2)**0.5
            dx = float(offset[1])/diag
            dy = float(offset[0])/diag#*(-1)
                        
            step_x = int(100*dx)
            step_y = int(100*dy)
            
            # print ('step_x = '+str(step_x)+' step_y = '+ str(step_y))
            
            
            cmd = '1pr'+str(int(step_x))
            self.command(cmd)
            time.sleep(0.2)
            
            cmd = '2pr'+str(int(step_y))
            self.command(cmd)
            time.sleep(0.2)
            
            current_litho_matrix = self.get_matches() 
            while np.count_nonzero(current_litho_matrix) < 1:
                current_litho_matrix = self.get_matches()
                
            offset = [0,0]
            lithocount = 0
            
            for row in range(len(current_litho_matrix)):
                for col in range(len(current_litho_matrix)):
                    if current_litho_matrix[row][col][0] !=0 and current_litho_matrix[row][col][1] != 0 :
                        lithocount += 1
                        offset[0] += current_litho_matrix[row][col][0] - self.LITHO_MATRIX[row][col][0] 
                        offset[1] += current_litho_matrix[row][col][1] - self.LITHO_MATRIX[row][col][1] 
                    
            offset[0] /= lithocount
            offset[1] /= lithocount
            
            offsets[0] = offsets[1]
            offsets[1] = offsets[2]
            offsets[2] = offset

            #if abs(offset[1])>abs(offsets[1][1]) and abs(offset[0])>abs(offsets[1][0]):
                #hint = C.windll.user32.MessageBoxW(0, "Motor direction seems suspicious!\n OK = continue adjusting current area\n Cancel = save current area and move on", "Help! I'm confused.", 1)
                #if hint == 2:
                #    break
                #else:
                    #dx = dx/2
                    #dy = dy/2

            print ('offsets = '+ str(offsets))
            
        if not initial_check:
            # here direction is used for incrementing coordinates
            direction = direction*2 - 1
            self.moved_in_X_not_Y = bool(moving_axis)
            if self.moved_in_X_not_Y:
                self.posx += direction
                
                if self.posx > self.maxx:
                    self.posx = 1
                    self.posX += 1
                elif self.posx == 0:
                    self.posx = self.maxx
                    self.posX -= 1
            else:
                self.posy += direction
                if self.posy > self.maxy:
                    self.posy = 1
                    self.posY += 1
        return
        
    
    # can be think of as a 3 by 3 by 2 tensor
    def get_matches(self, threshold = 0.7):
        gray_img = self.crop_image()
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        error_tolerance = int(0.5*(self.MAX_LITHO_Y-self.MIN_LITHO_Y))
        #gray_img = cv2.flip(gray_img,0)
        
        res = cv2.matchTemplate(gray_img,self.template,cv2.TM_CCOEFF_NORMED)
        
        loc = np.where( res >= threshold)
        matches = []
        for pt in zip(*loc[::-1]):
            distinct_pt = True
            for match in matches:
                if abs(pt[0]-match[0])<(self.unitsq/3.5) and abs(pt[1]-match[1])<(self.unitsq/3.5):
                    distinct_pt = False
                    break
            if (distinct_pt):
                matches.append(pt) 
                
        matches = sorted(matches, key=itemgetter(1))
        print (matches)
        
        lithomarks =  np.zeros_like(self.LITHO_MATRIX)
        
        if self.obj_mag == 100:
            centre = (self.MAX_LITHO_X + self.MIN_LITHO_X)/2
        
        for match in matches:
            if abs(match[0]-self.MIN_LITHO_X)<error_tolerance and abs(match[1]-self.MIN_LITHO_Y)<error_tolerance:
                lithomarks[0][0] = match
            if self.obj_mag == 100 and abs(match[0]-self.MIN_LITHO_X)<error_tolerance and abs(match[1]-centre)<error_tolerance:
                lithomarks[0][1] = match
            if abs(match[0]-self.MIN_LITHO_X)<error_tolerance and abs(match[1]-self.MAX_LITHO_Y)<error_tolerance:
                lithomarks[0][(len(lithomarks[0])-1)] = match
                
            if self.obj_mag == 100 and abs(match[0]-centre)<error_tolerance and abs(match[1]-self.MIN_LITHO_Y)<error_tolerance:
                lithomarks[1][0] = match 
            if self.obj_mag == 100 and abs(match[0]-centre)<error_tolerance and abs(match[1]-centre)<error_tolerance:
                lithomarks[1][1] = match
            if self.obj_mag == 100 and abs(match[0]-centre)<error_tolerance and abs(match[1]-self.MAX_LITHO_Y)<error_tolerance:
                lithomarks[1][(len(lithomarks[0])-1)] = match
                
            if abs(match[0]-self.MAX_LITHO_X)<error_tolerance and abs(match[1]-self.MIN_LITHO_Y)<error_tolerance:
                lithomarks[(len(lithomarks[0])-1)][0] = match
            if self.obj_mag == 100 and abs(match[0]-self.MAX_LITHO_X)<error_tolerance and abs(match[1]-centre)<error_tolerance:
                lithomarks[(len(lithomarks[0])-1)][1] = match
            if abs(match[0]-self.MAX_LITHO_X)<error_tolerance and abs(match[1]-self.MAX_LITHO_Y)<error_tolerance:
                lithomarks[(len(lithomarks[0])-1)][(len(lithomarks[0])-1)] = match
        
        if np.count_nonzero(lithomarks) < 1:
            pass
            #hint = C.windll.user32.MessageBoxW(0, "Off the grids / Out of focus / Rotated too much / Grids are contaminated \nOK = help me move with WASD\nCancel = ignore and move on", "Help! I'm in trouble", 1)
            #if hint == 2:
            #        return self.LITHO_MATRIX
            #self.User_Preparation()
            
        return lithomarks

            
    # a greater variance of Laplacian indicates stronger edges and thus, better focus
    # replace Laplacian with Sobel operators for direction-specific edge detection
    # returns maximum variance ('the clearest parameter')
    def AutoFocus_Variance (self, coarseness=1, saveSnapshot=True):
        self.command('st')
        cropped_gray_img = self.crop_image()
        variance = [cv2.Laplacian(cropped_gray_img, cv2.CV_64F).var(),0,0]
        #print (variance)
        
        '''
        # TODO? rotate between +z / -z for severely out of focus cases that needs many iterations
        '''
        cointoss = bool(random.getrandbits(1))
        default_direction = '-' if (cointoss) else ('')
        
        
        # coarseness is determined by how bad the field looks in comparison to the best one so far
        step_size = 10*coarseness
        
        # adjust towards default direction
        cmd = '3pr'+default_direction+str(step_size)
        self.command(cmd)
        time.sleep(0.005*coarseness)
        #print ('First move towards '+default_direction+' z direction')
        cropped_gray_img = self.crop_image()
        variance[1] = cv2.Laplacian(cropped_gray_img, cv2.CV_64F).var()
        print ('variance --- '+str(variance))
        # if edges gets stronger as we go towards +z, the direction is +
        #direction = '' if ((variance[1] > variance[0] and default_direction == '') or (variance[1] < variance[0] and default_direction == '-') ) else ('-')
        neg_direction = '-' if (default_direction == '' ) else ('')
        direction = ''
        
        # if default_direction makes things worse, we may need to move towards the other direction or just 0
        if (variance[1] < variance[0]):
            cmd = '3pr'+neg_direction+str(2*step_size)
            self.command(cmd)
            time.sleep(0.01*coarseness)
            self.Camera.SnapImage()
            cropped_gray_img = self.crop_image()
            variance[2] = cv2.Laplacian(cropped_gray_img, cv2.CV_64F).var()
            if ((variance[1] < variance[0]) and (variance[2] < variance[0])):
                # origin is better than either direction - back to origin
                cmd = '3pr'+default_direction+str(step_size)
                self.command(cmd)
                time.sleep(0.005*coarseness)
                self.crop_image(save=saveSnapshot,analyze=bool(saveSnapshot and coarseness==1))
                if (variance[0]>self.autofocus_reference):
                    self.autofocus_reference = variance[0]
                self.set_home_position()
                return variance[0]
            else:
                # if we need to keep going towards -z, var[2]>var[0]>var[]
                variance[1] = variance[2]
                direction = neg_direction
                neg_direction = default_direction
        else:
            direction = default_direction
                
                
        # now we know which direction to go 
        while (variance[1]>variance[0] or variance[2]>=variance[1]):
            cmd = '3pr'+direction+str(step_size)
            self.command(cmd)
            time.sleep(0.005*coarseness)
            self.Camera.SnapImage()
            cropped_gray_img = self.crop_image()

            variance[2] = cv2.Laplacian(cropped_gray_img , cv2.CV_64F).var()
            
            if ((variance[1] - variance[0])*(variance[2] - variance[1]) < 0):
                # d/dz changes sign = extremum = focused, half a step back.
                cmd = '3pr'+neg_direction+str(int(step_size/2))
                self.command(cmd)
                time.sleep(0.005*coarseness)
                self.crop_image(save=saveSnapshot,analyze=bool(saveSnapshot and coarseness==1))
                if (max(variance)>=self.autofocus_reference):
                    self.autofocus_reference = max(variance)
                self.set_home_position()
                break
            
            variance[0] = variance[1]
            variance[1] = variance[2] 
            variance[2] = 0
            
        return variance[1]
    
    # crop image and save it (or not)
    def crop_image (self,save=False,FindNumberMatches=False, analyze=False):
        self.Camera.SnapImage()
        bayer = self.Camera.GetImageEx()
        if not FindNumberMatches:
            bayer = bayer[:, 360:-360 ]
        #image = cv2.flip(image,0)
        image = cv2.cvtColor(bayer, 49)
        #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #raw_w, raw_h = image.shape[::-1]
        #ybd = int((raw_h-height)/2)
        #xbd = int((raw_w-width)/2)
        #image = image[ybd:-ybd, xbd:-xbd]
        if save:
            #self.last_central_square = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            flakename = 'X'+str(self.posX)+'x'+str(self.posx)+'_'+'Y'+str(self.posY)+'y'+str(self.posy) + '.png'
            bayername = flakename[:-4]+'.tif'
            print (bayername)
            cv2.imwrite(os.path.join(self.out_path , flakename),image)
            cv2.imwrite(os.path.join(self.out_path_bayer , bayername),bayer)
            if analyze:
                self.analyze(flakename)
            
        return image
    
    def analyze (self,sample):
        steps = ['', '', '1_Image_Acquisition','Global_Histogram','2_Segmentation', '3_ROIs', '4_Local_Histogram']
        full_path = []
        for step in steps:
            full_path.append(os.path.join(self.result_path,step))
            
        self.ContourObj = Contour.Contour(str(sample), full_path[2], self.result_path, obj_mag=self.obj_mag, material = self.material,ftr=self.filter)
        if self.material == 'Sb':
            self.ContourObj.edges_gluey(obj_mag=self.obj_mag, presentation=self.presentation)
        else:
            self.ContourObj.edges_gluey(obj_mag=self.obj_mag, ftr=self.filter, glue_remover=False, presentation=self.presentation)
        #self.ContourObj.segmentation(ftr=self.filter)
        self.ROIObj = Rectangle.Rectangle((full_path[4]+'\\Contours'), sample, full_path[2], self.result_path, obj_mag=self.obj_mag, material=self.material,ftr=self.filter)
        X_Y_FS = self.ROIObj.markROIs(save=True, presentation=self.presentation)
        
        list_of_chunks = os.listdir((full_path[5]+'\\'+sample[:-4]))
        list_of_chunk_numbers = []
        if self.material == 'C':  
            for chunk,ant in zip(list_of_chunks,X_Y_FS):
                stat = str(sample)[:-4]+'_'
                self.HistogramObj = Histogram.Histogram(str(chunk), (full_path[5]+'\\'+sample[:-4]), (full_path[6]), self.result_path,obj_mag=self.obj_mag)
                #self.HistogramObj = Histogram.Histogram(str(chunk), (full_path[5]+'\\'+sample), (full_path[6]+'\\'+sample), self.result_path,obj_mag=self.obj_mag) #deeper directory
                contrast,layers = self.HistogramObj.saveLocalHistogram(ftr=self.filter, material=self.material, presentation=self.presentation)
                if not contrast is None and len(contrast)>3:
                    chunk_number = int(chunk[-6:-4])
                    stat += str(chunk_number).zfill(2) + ',' + contrast
                    stat +=  ',' + layers
                    list_of_chunk_numbers.append(chunk_number)
                if ant[3] in list_of_chunk_numbers:
                    stat += ","+str(ant[2]) +"\n"
                    stats_file = open(os.path.join(self.result_path, 'ListOfFlakes.csv') ,"a+")
                    stats_file.write(stat)
                    stats_file.close()
        cv2.destroyAllWindows()

    def set_home_position (self):
        self.command('1dh')
        self.command('2dh')
        self.command('3dh')
        
        
    '''
    def User_Preparation (self):
        pass
        self.Camera.StopLive()
        watermark = cv2.imread(os.path.join( 'LithoExamplar',str(self.obj_mag) ,'cross.png'),-1)
        watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.unitsq)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.unitsq)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            frame_h, frame_w, frame_c = frame.shape
            # overlay with 4 channels BGR and Alpha
            overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
            watermark_h, watermark_w, watermark_c = watermark.shape
            # replace overlay pixels with watermark pixel values
            offseth = int((frame_h-watermark_h)/2)
            offsetw = int((frame_w-watermark_w)/2)
            for i in range(0, watermark_h):
                for j in range(0, watermark_w):
                    if watermark[i,j][3] != 0:
                        h_offset = frame_h - watermark_h - offseth
                        w_offset = frame_w - watermark_w - offsetw
                        overlay[h_offset + i, w_offset+ j] = watermark[i,j]
            cv2.addWeighted(overlay, 0.25, frame, 1.0, 0, frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            cv2.imshow('ALIGN and FOCUS (press SPACE to continue)',frame)
            # guarding against non-orthogonal placement
            if cv2.waitKey(20) & 0xFF == ord('a'): 
                self.command('2pr50')
                time.sleep(0.1)
            if cv2.waitKey(20) & 0xFF == ord('d'): 
                self.command('2pr-50')
                time.sleep(0.1)
            if cv2.waitKey(20) & 0xFF == ord('w'): 
                self.command('1pr-50')
                time.sleep(0.1)
            if cv2.waitKey(20) & 0xFF == ord('s'): 
                self.command('1pr50')
                time.sleep(0.1)
            if cv2.waitKey(20) & 0xFF == ord(' '): 
                #self.last_central_square = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.destroyAllWindows()
                cap.release()
                break
        self.Camera.StartLive(1)
        return
    '''
    
    
if __name__ == '__main__':
    BayerAcq(os.getcwd())