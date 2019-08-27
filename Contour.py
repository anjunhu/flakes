from __future__ import print_function
import os
import cv2
import argparse
import copy
import numpy as np
from scipy.ndimage import label
import scipy
from collections import Counter
import random
import statistics

class Contour:
    
    def __init__(self, fn_original, src_path, out_path, material='C',obj_mag=50, ftr=542):
        self.fn_original = fn_original
        self.path_original = src_path+'\\'+fn_original
        self.out_path_contour = out_path+'\\2_Segmentation\\Contours'
        if not os.path.exists(self.out_path_contour):
            os.makedirs(self.out_path_contour)
        self.out_path_outline = out_path+'\\2_Segmentation\\Outline'
        
        if ftr<600:
            self.out_path_BrightFlakes = out_path+'\\BrightFlakes'
            if not os.path.exists(self.out_path_BrightFlakes):
                os.makedirs(self.out_path_BrightFlakes)
        self.material = material
        self.error_margin = 50
        self.cross = cv2.imread(os.path.join('LithoExamplar\\50' ,'cross.png'),0)
        ret, thresh = cv2.threshold(self.cross, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,1)
        #print(contours)
        self.crosscontour = contours[0]
        
        
        self.obj_mag = obj_mag
        LithoExamplarDir = os.path.join( 'LithoExamplar',str(self.obj_mag))    
        self.lithomask0 = cv2.imread(os.path.join(LithoExamplarDir, 'lithomask.png') ,0)
        self.template = cv2.imread(os.path.join(LithoExamplarDir ,'cross.png'),0)
        self.coords = []
        self.coords.append(self.template)
        for i in range(1,10):
            self.template = cv2.imread(os.path.join(LithoExamplarDir,(str(i)+'.png')),0)
            self.coords.append(self.template)
        self.error_margin = 50

        
    
    # litho-markers make up 4% of the total area
    # for G/R: ND16 out
    def edges_gluey(self, obj_mag=50, ftr=0, glue_remover=True,polygonareaestimate=True, presentation=1):
        
        original = cv2.imread(self.path_original)  
        #original = cv2.bilateralFilter(original,9,75,75)
        original = cv2.medianBlur(original,5)  # noise reduction
 
        img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        mode = statistics.mode(img_gray.flatten())
            
        light_litho = np.array([100, 255, 255])
        dark_litho = np.array([50, max(200,min(int(mode*3),240)), max(200,min(int(mode*3),240))])
        
        if self.material == 'C':
            light_litho = np.array([150, 255, 255])
            
        if presentation and self.material=='Sb':
            win1 = str('Original '+self.fn_original[:-4])
            cv2.namedWindow(win1)      
            cv2.moveWindow(win1, 0,0)
            cv2.imshow(win1, cv2.resize(original, (768, 768)))
            key = cv2.waitKey(1000)
            if key == 27: #ESC
                cv2.destroyAllWindows()
        
        
        
        b, g, r = cv2.split(original)
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        
        
        if ftr==0:      #ND
            lithomask = cv2.inRange(original, dark_litho, light_litho)
        elif ftr<500:   #B
            img_gray = b
        elif ftr>600:   #R
            #gray_thres = np.percentile(g.flatten(),98)
            coefficients = [0,0.4/1.3,0.9/1.3] 
            m = np.array(coefficients).reshape((1,3))
            img_gray = cv2.transform(original, m)
            light_litho = np.array([255, 255, 255])
            dark_litho = np.array([0, 160, 250])
            lithomask = cv2.inRange(original, dark_litho, light_litho)
        else:           #G
            img_gray = g
            light_litho = np.array([255, 255, 255])
            dark_litho = np.array([125, 250, 40])
            lithomask = cv2.inRange(original, dark_litho, light_litho)
        
        #cv2.imwrite(str(self.fn_original[:-4]+'img_gray.png'), img_gray)
        
     
        
        
        lithomask = cv2.dilate(lithomask,kernel,iterations = 10)
        #cv2.imshow('lithomask', lithomask)
        #cv2.waitKey(0)
        lithomask2 = np.zeros_like(img_gray)
        _, brightest_mask = cv2.threshold(img_gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        #img_gray_thres = np.percentile(img_gray.flatten(),99)
        #_, brightest_mask = cv2.threshold(img_gray, img_gray_thres, 255, cv2.THRESH_BINARY)
        
        '''
        lithocontour_list,hierarchy = cv2.findContours(brightest_mask,cv2.RETR_EXTERNAL,1)
        for cnt in lithocontour_list:
            for i in range(0,10):
                matchscore = cv2.matchShapes(cnt,self.coords[i],1,0.0)
                #print(matchscore)
                if  matchscore < 0.1:
                    cv2.drawContours(lithomask2, [cnt], 0, 255, cv2.FILLED)
                
          
        lithomask2 = cv2.dilate(lithomask2,kernel,iterations = 10) 
        
        
        # if using matchTemplate, dilate before adding lithomask2
        for i in range(0,10):
            w_temp, h_temp = self.coords[i].shape[::-1] 
            res = cv2.matchTemplate(img_gray,self.coords[i],cv2.TM_CCOEFF_NORMED)
            threshold = 0.65
            #mask.fill(255)
            loc = np.where( res >= threshold)
            matches = []

            for pt in zip(*loc[::-1]):
                distinct_pt = True
                for match in matches:
                    if abs(pt[0]-match[0])<5*self.obj_mag and abs(pt[1]-match[1])<5*self.obj_mag:
                        distinct_pt = False
                        break
                if (distinct_pt and i==0):
                    pass
                    #cv2.rectangle(lithomask2, (pt[0]+40, pt[1]), (pt[0]+120, pt[1]+180), 255, -1)
                    #cv2.rectangle(lithomask2, (pt[0], pt[1]+40), (pt[0]+180, pt[1]+120), 255, -1)
                #elif distinct_pt:
                    #cv2.rectangle(lithomask2, (pt[0]-10, pt[1]-10), (pt[0]+90, pt[1]+180), 255, -1) 
                    

        
            

        '''
        central_sq_mask = np.zeros_like(img_gray)
        
        if obj_mag==50 and self.material=='Sb':
            cv2.rectangle(central_sq_mask, (300, 0), (900, 1200), 255, -1)
            cv2.rectangle(central_sq_mask, (0,400), (1200, 1000), 255, -1)
            pt = None
            if self.fn_original[3]=='1' and self.fn_original[8]=='1':
                pt = (0,0)
            if self.fn_original[3]=='1' and self.fn_original[8]=='5':
                pt = (850,0)
            if self.fn_original[3]=='5' and self.fn_original[8]=='1':
                pt = (0,840)
            if self.fn_original[3]=='5' and self.fn_original[8]=='5':
                pt = (850,840)
            if not pt is None:
                cv2.rectangle(central_sq_mask, (pt[0],pt[1]), (pt[0]+400, pt[1]+200), 0, -1)
                
        lithomask = cv2.bitwise_or(lithomask, lithomask2)
        lithomask = cv2.bitwise_not(lithomask)
        lithomask = cv2.bitwise_or(lithomask, central_sq_mask)
        #cv2.imwrite(str('lithomask_'+self.fn_original), lithomask)
        
        '''
        b, g, r = cv2.split(original)
        
        w, h = img_gray.shape[::-1]
        
        '''

        '''    
        brightest=255
        expected_bright_area = 0.005*w*h
        accumulated_litho_area = 0
        histr = cv2.calcHist([g],[0],None,[256],[0,256])
        histr = histr.ravel()
        while accumulated_litho_area < expected_bright_area:
            accumulated_litho_area += histr[brightest]
            brightest -= 1
        '''
        
        '''
        gray_thres = np.percentile(img_gray.flatten(),98)
        _, brightest_mask = cv2.threshold(img_gray, gray_thres, 255, cv2.THRESH_BINARY)
        brightest_mask = cv2.bitwise_and(brightest_mask,lithomask)
        '''
        #brightest_mask = cv2.bitwise_not(brightest_mask)
        #brightest_mask = cv2.bitwise_or(brightest_mask,central_sq_mask)
        #cv2.imwrite(os.path.join(self.out_path_contour , str('brightest_mask_'+self.fn_original)), brightest_mask) 
        
        
        # for 10x objective: only take the upper-left 100 by 100 um area
        #w, h = img_gray.shape[::-1]
        if obj_mag==10:
            img_gray = img_gray[:int(h_big*0.6),:int(w_big*0.6)]
            original = original[:int(h_big*0.6),:int(w_big*0.6)]
            
            
        w_big, h_big = img_gray.shape[::-1]
        
        sure_flakes_counter = 1
        
        glue_mask = np.zeros_like(img_gray)
        glue_mask = cv2.bitwise_not(glue_mask)
        
        brightest_mask = cv2.bitwise_and(brightest_mask,lithomask)
        contours, _ = cv2.findContours(brightest_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            isGlue = False
            x,y,w,h = cv2.boundingRect(cnt)
            region = original[y:int(y+h*1), x:int(x+w*1)]
                
            if glue_remover:
                b, g, r = cv2.split(region)
                #Y, Cr, Cb = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb))
                H, L, S = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2HLS))
                
                regionlithomask = cv2.inRange(region, dark_litho, light_litho)
                regionlithomask = cv2.dilate(regionlithomask,kernel,iterations = 3)
                
                L[regionlithomask>0] = np.median(L)
                Lvar = S.var()
                S[regionlithomask>0] = np.median(S)
                Svar = S.var()
                #print('Svar = '+str(Svar))
                #cv2.imshow('S', S)
                #cv2.waitKey(0)
                #isGlue = bool(Lvar<800) or bool(Svar<2500)
        
                isGlue = bool(Lvar<450) or bool(Svar<1400)
        
                
                '''
                H[regionlithomask>0] = np.median(H)
                Hvar = H.var()
                #print('Hvar = '+str(Hvar))
                #cv2.imshow('H', H)
                #cv2.waitKey(0)
                isGlue = (isGlue) or bool(Hvar > 100)
                #isGlue = False
                '''
                
                
                '''            
                r[regionlithomask>0] = np.median(r)
                g[regionlithomask>0] = np.median(g)
                r_thres = np.percentile(r.flatten(),98)
                g_thres = np.percentile(g.flatten(),98)
                _, r_bin = cv2.threshold(r, r_thres, 255, cv2.THRESH_BINARY)
                _, g_bin = cv2.threshold(g, g_thres, 255, cv2.THRESH_BINARY) # another lowerbound = g_mode[0]
                r_area = np.sum(r_bin)/255
                g_area = np.sum(g_bin)/255
                rg_overlap = cv2.bitwise_and(r_bin, g_bin)
                #rg_overlap = r_bin-g_bin
    
                rg_overlap_area = np.sum(rg_overlap)/255
                overlapscore = 0
                if rg_overlap_area>0:
                    overlapscore = rg_overlap_area/max(r_area,g_area)
                    
                isGlue = overlapscore < 0.7
      
                '''
    
            if isGlue:
                cv2.drawContours(glue_mask, [cnt], 0, 0, cv2.FILLED)
            elif self.material=='Sb':
                #print(Svar)
                #print(darkest)
                y_lower = max(0,int(y-h*0.2))
                y_upper = min(int(y+h*1.2),h_big-1)
                x_lower = max(0,int(x-w*0.2))
                x_upper = min(int(x+w*1.2),w_big-1)
                region = original[y_lower:y_upper,x_lower:x_upper]
                if presentation and self.material=='Sb':
                    win1 = str('Bright Flake Detected!! in area '+self.fn_original[:-4])
                    cv2.namedWindow(win1)      
                    cv2.moveWindow(win1, 780,0)
                    displayw = int(min(650,5*w))
                    displayh = int(displayw/w*h)
                    cv2.imshow(win1, cv2.resize(region, (displayw,displayh)))
                    key = cv2.waitKey(1000)
                    if key == 27: #ESC
                        cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(self.out_path_BrightFlakes , (self.fn_original[:-4]+'_'+str(sure_flakes_counter).zfill(2)+'.png')), region) 
                sure_flakes_counter += 1
                #if self.material == 'C' and ftr>580:
                #    cv2.drawContours(glue_mask, [cnt], 0, 0, cv2.FILLED)
            
        
        # take discrete laplacian and binarize
        laplacian = cv2.Laplacian(img_gray,cv2.CV_64F)
        laplacian[laplacian <= 2 ] = 0
        laplacian[laplacian > 2 ] = 255  
        
        # dilate and erode to protect little flakes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        laplacian = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, kernel,iterations = 1)
        
        
        
        cv2.imwrite(os.path.join(self.out_path_contour , self.fn_original), laplacian)


        if presentation and self.material=='C':
            win1 = str('Binarized Laplacian for '+self.fn_original[:-4])
            cv2.namedWindow(win1)      
            cv2.moveWindow(win1, 0,0)
            cv2.imshow(win1, cv2.resize(laplacian, (768, 768)))
            key = cv2.waitKey(1000)
            if key == 27: #ESC
                cv2.destroyAllWindows()
            '''
            win2 = str('Convex Polygons and Convex Defects for '+self.fn_original)
            cv2.namedWindow(win2)      
            cv2.moveWindow(win2, 500,0)
            cv2.imshow(win2,  cv2.resize(original, (512, 512)))
            key = cv2.waitKey(1000)
            if key == 27: #ESC
                cv2.destroyAllWindows()
            '''
                            
        # throw away noise
        laplacian = cv2.imread(os.path.join(self.out_path_contour , self.fn_original), 0)
        laplacian = cv2.bitwise_and(laplacian, lithomask)
        contours, _ = cv2.findContours(laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        binary_map = (laplacian > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, 4, cv2.CV_32S)
        labels_flat = labels.flatten()
        counter = Counter(labels_flat).most_common()
        #print(str(labels))
        
        
        laplacian = np.zeros_like(laplacian)
        for lbl in counter:
            if (lbl[1] < obj_mag*4 and self.material=='C') or (lbl[1] < obj_mag*4 and self.material=='Sb'): 
                break
            elif lbl[0] != 0:
                
                laplacian[labels==lbl[0]] = 255

        laplacian_dil = cv2.dilate(laplacian,kernel,iterations = 2)
        
        

        contours, _ = cv2.findContours(laplacian_dil,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        polygonareaestimate = polygonareaestimate or self.material=='Sb'                             
        if polygonareaestimate:                             
            #flood_start_point = (0,0)
            
            mode = 1
            modeloopcounter = 0
            while mode > 0 and modeloopcounter<5 :
                flood_start_point = (random.randint(0,w_big-1),random.randint(0,h_big-1))
                #print('mode = '+str(mode)+', new point of choice: '+str(flood_start_point))
                holes = laplacian_dil.copy()
                cv2.floodFill(holes, None, flood_start_point, 255)
                holes = cv2.bitwise_not(holes)
                filled_laplacian = cv2.bitwise_or(laplacian_dil, holes)
                mode = statistics.mode(filled_laplacian.flatten())
                modeloopcounter += 1
                
                
            neg_laplacian = cv2.bitwise_not(laplacian)
            solid_laplacian = cv2.bitwise_and(filled_laplacian, neg_laplacian)
            
            
            
            glue_mask = cv2.erode(glue_mask,kernel,iterations = 2)
            #cv2.imwrite(str('gluemask_'+self.fn_original), glue_mask)
            solid_laplacian = cv2.bitwise_and(solid_laplacian, glue_mask)
            solid_laplacian = cv2.bitwise_or(solid_laplacian, brightest_mask)

        
        
            contours, _ = cv2.findContours(solid_laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            region = original[y:int(y+h*1), x:int(x+w*1)]
            isGlue = False #TESTFORGRAPHENE
            if glue_remover:
                b, g, r = cv2.split(region)
                #Y, Cr, Cb = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb))
                H, L, S = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2HLS))
                regionlithomask = cv2.inRange(region, dark_litho, light_litho)
                regionlithomask = cv2.dilate(regionlithomask,kernel,iterations = 3)
                S[regionlithomask>0] = np.median(S)
                Svar = S.var()
                #print('Svar = '+str(Svar))
                #cv2.imshow('S', S)
                #cv2.waitKey(0)
                isGlue = bool(Svar<1000)
              
                r[regionlithomask>0] = np.median(r)
                g[regionlithomask>0] = np.median(g)
                r_thres = np.percentile(r.flatten(),98)
                g_thres = np.percentile(g.flatten(),98)
                _, r_bin = cv2.threshold(r, r_thres, 255, cv2.THRESH_BINARY)
                _, g_bin = cv2.threshold(g, g_thres, 255, cv2.THRESH_BINARY) # another lowerbound = g_mode[0]
                r_area = np.sum(r_bin)/255
                g_area = np.sum(g_bin)/255
                rg_overlap = cv2.bitwise_and(r_bin, g_bin)
                #rg_overlap = r_bin-g_bin
    
                rg_overlap_area = np.sum(rg_overlap)/255
                overlapscore = 0
                if rg_overlap_area>0:
                    overlapscore = rg_overlap_area/max(r_area,g_area)
                    
                isGlue = isGlue or overlapscore < 0.7
                
                
                '''
                H[regionlithomask>0] = np.median(H)
                Hvar = H.var()
                #print('Hvar = '+str(Hvar))
                #cv2.imshow('H', H)
                #cv2.waitKey(0)
                isGlue = (isGlue) or bool(Hvar > 100)
                '''
                isGlue = (isGlue) and (bool(np.percentile(L.flatten(),0.5)>65))
                
            if isGlue:
                cv2.drawContours(solid_laplacian, [cnt], 0, 0, cv2.FILLED)
            #else:
                #print(Svar)
  
        '''
        glue_mask = cv2.erode(glue_mask,kernel,iterations = 2)
        #cv2.imwrite(str('gluemask_'+self.fn_original), glue_mask)
        solid_laplacian = cv2.bitwise_and(solid_laplacian, glue_mask)
        solid_laplacian = cv2.bitwise_or(solid_laplacian, brightest_mask)
        '''
        

        if polygonareaestimate: 
            contours, _ = cv2.findContours(solid_laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            
            litho_hull_list = []
            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)
                cv2.drawContours(original, contours, i, (250,250,250))
                x,y,w,h = cv2.boundingRect(cnt)
                hull = cv2.convexHull(cnt, returnPoints=False)
                defects = cv2.convexityDefects(cnt,hull)
                #defects = sorted(defects, key=defects[3], reverse=True)
                if not defects is None:
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        cv2.line(original,start,end,[255,255,255],1)
                        cv2.circle(original,far,2,[0,0,255],-1)
          
                    hull = cv2.convexHull(cnt)
                    litho_hull_list.append(hull)
                    cv2.drawContours(solid_laplacian, [hull], 0, 255, cv2.FILLED)
                    
            #cv2.imshow('Contours', drawing)
            #cv2.waitKey(0) 

            
            #self.out_path_contour = self.out_path_contour+'\\temp'
            cv2.imwrite(os.path.join(self.out_path_contour , self.fn_original), solid_laplacian)
            if not os.path.exists(self.out_path_outline):
                os.makedirs(self.out_path_outline)
            #original[laplacian == 255] = (0, 0, 255)
            cv2.imwrite(os.path.join(self.out_path_outline , self.fn_original), original)
            if presentation and self.material=='C':
                cv2.imshow(win1, cv2.resize(solid_laplacian, (768, 768)))
                key = cv2.waitKey(1000)
                if key == 27: #ESC
                    cv2.destroyAllWindows()
        
        return
            

    
    def segmentation(self, ftr=0):
        original = cv2.imread(self.path_original)  
        # Further noise reduction
        original = cv2.GaussianBlur(original,(5,5),0)
        # Convert to grayscale and binary
        img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  
        
        if ftr==0:      #ND
            pass
        elif ftr<500:   #B
            img_gray, _, _ = cv2.split(original)
        elif ftr>600:   #R
             _, _, img_gray= cv2.split(original)
        else:           #G
            _, img_gray, _= cv2.split(original)
            
            
        img_gray_inv = 255 - img_gray
        contour = 0
        
        histr = cv2.calcHist([img_gray],[0],None,[256],[0,255])
        histr = histr.ravel()
        substrate = np.argmax(histr) 
        print(substrate)
        mask = np.zeros_like(img_gray)
        mask[(img_gray==substrate)] = 255
        mask = cv2.bitwise_not(mask)
        
        #cv2.imshow('',img_gray)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #_, img_bin_inv = cv2.threshold(img_gray_inv, 0, 255, cv2.THRESH_OTSU)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,np.ones((3, 3), dtype=int))
        # Boundaries that vaguely seperate flakes and substrate
        border = cv2.dilate(img_bin, None, iterations=2)
        border = border - cv2.erode(border, None)
        # Once we have a binary image and boundaries that divide the "flakes"
        # and the "substrate", we can calculate the distance from a point in the
        # "flake region" to its nearest "substrate region"
        dt = cv2.distanceTransform(img_bin, 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, 0, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        lbl[border == 255] = 255
        lbl = lbl.astype(np.int32)
        cv2.watershed(original, lbl)
        lbl[lbl == -1] = 0
        #print(lbl)
        #print ('label length '+str(len(lbl)))
        #print ('label[0] length '+str(len(lbl[0])))
        lbl = lbl.astype(np.uint8)
         
        contour = 255 - lbl
        
        
        result = copy.deepcopy(contour)
        # Encircle the flakes on the original image. Not really useful for the
        # coming steps but can be used as an intuitive sanity check
        #result[result != 255] = 0
        #result = cv2.dilate(result, None)
        original[result == 255] = (0, 0, 255)
        cv2.imwrite(os.path.join(self.out_path_contour , self.fn_original), contour)
        cv2.imwrite(os.path.join(self.out_path_outline , self.fn_original), original)
        return [contour, original]
    
            

    # litho-markers make up 4% of the total area
    def edges(self, obj_mag=50, ftr=0, presentation=1):
        original = cv2.imread(self.path_original)  
        #original = cv2.bilateralFilter(original,9,75,75)
        original = cv2.medianBlur(original,5)  # noise reduction
    
    
        # use appropriate 8-bit monochrome image based on one's choice of filter
        img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) 
        blurry_brightness = cv2.GaussianBlur(img_gray, (11, 11), 0)
        b, g, r = cv2.split(original)
        if ftr==0:      #ND
            pass
        elif ftr<500:   #B
            img_gray = b
        elif ftr>600:   #R
            img_gray= r
        else:           #G
            img_gray = g
            
        
        w, h = img_gray.shape[::-1]
        central_sq_mask = np.zeros_like(img_gray)
        if obj_mag==50:
            cv2.rectangle(central_sq_mask, (250, 350), (1000, 870), 255, -1)
            #cv2.imshow('central_sq_mask', central_sq_mask)
            #cv2.waitKey(0)
            
        brightest=255
        expected_litho_area = 0.005*w*h
        accumulated_litho_area = 0
        histr = cv2.calcHist([blurry_brightness],[0],None,[256],[0,256])
        histr = histr.ravel()
        while accumulated_litho_area < expected_litho_area:
            accumulated_litho_area += histr[brightest]
            brightest -= 1
        
        
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        brightest_mask = np.zeros_like(img_gray)
        brightest_mask[img_gray>brightest] = 255
        brightest_mask = cv2.dilate(brightest_mask,kernel,iterations = 10)
        brightest_mask = cv2.bitwise_not(brightest_mask)
        brightest_mask = cv2.bitwise_or(brightest_mask,central_sq_mask)
        #cv2.imwrite(os.path.join(self.out_path_contour , str('brightest_mask_'+self.fn_original)), brightest_mask) 
        

        

        
        # for 10x objective: only take the upper-left 100 by 100 um area
        #w, h = img_gray.shape[::-1]
        if obj_mag==10:
            img_gray = img_gray[:int(h*0.6),:int(w*0.6)]
            original = original[:int(h*0.6),:int(w*0.6)]
            w, h = img_gray.shape[::-1]
        
        # take discrete laplacian and binarize
        laplacian = cv2.Laplacian(img_gray,cv2.CV_64F)
        laplacian[laplacian <= 2 ] = 0
        laplacian[laplacian > 2 ] = 255  
        
        # dilate and erode to protect little flakes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        laplacian = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, kernel,iterations = 1)
        cv2.imwrite(os.path.join(self.out_path_contour , self.fn_original), laplacian)
        # throw away noise
        laplacian = cv2.imread(os.path.join(self.out_path_contour , self.fn_original), 0)
        laplacian = cv2.bitwise_and(laplacian, brightest_mask)
        contours, _ = cv2.findContours(laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        binary_map = (laplacian > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, 4, cv2.CV_32S)
        labels_flat = labels.flatten()
        counter = Counter(labels_flat).most_common()
        #print(str(labels))
        
        
        laplacian = np.zeros_like(laplacian)
        for lbl in counter:
            if lbl[1] < obj_mag*5: 
                break
            elif lbl[0] != 0:
                
                laplacian[labels==lbl[0]] = 255
                
        #flood_start_point = (0,0)
        mode = 1
        contours, _ = cv2.findContours(laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        while mode > 0:
            flood_start_point = (random.randint(0,w-1),random.randint(0,h-1))
            #print('mode = '+str(mode)+', new point of choice: '+str(flood_start_point))
            holes = laplacian.copy()
            cv2.floodFill(holes, None, flood_start_point, 255)
            holes = cv2.bitwise_not(holes)
            filled_laplacian = cv2.bitwise_or(laplacian, holes)
            mode = statistics.mode(filled_laplacian.flatten())
            
        '''            
        contours, _ = cv2.findContours( filled_laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)            
        hull_list = []
 
        for i in range(len(contours)):
            hul = cv2.convexHull(contours[i], False)
            hull_list.append(hul)
            defect = cv2.convexityDefects(contours[i],hul)
            if len(defect)==4:
                print(str(contours[i]))
                #contours.remove(contours[i])
        '''   
        
        contours, _ = cv2.findContours(filled_laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        
        '''
        for cnt in contours:
            #cnt = cv2.approxPolyDP(cnt,0.1,True)
            
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            if (perimeter/area)<0.1 and abs(area-3.8*obj_mag**2)<500:
                #print('area = '+str(area)+', perim = '+str(perimeter)+', convex = '+str(perimeter/area))
                #cv2.imshow('Contours', original)
                #cv2.waitKey(0)
                cv2.drawContours(original, [cnt], 0, (0,0,250))
            else:
                cv2.drawContours(original, [cnt], 0, (250,250,250))
 
            
        '''
        
        litho_hull_list = []
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            cv2.drawContours(original, contours, i, (250,250,250))
            x,y,w,h = cv2.boundingRect(cnt)
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt,hull)
            #defects = sorted(defects, key=defects[3], reverse=True)
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(original,start,end,[255,255,255],1)
                cv2.circle(original,far,2,[0,0,255],-1)
  
            hull = cv2.convexHull(cnt)
            litho_hull_list.append(hull)
            cv2.drawContours(filled_laplacian, [hull], 0, 255, cv2.FILLED)
        #cv2.imshow('Contours', drawing)
        #cv2.waitKey(0) 
        
        cv2.imwrite(os.path.join(self.out_path_contour , self.fn_original), filled_laplacian)
        
        #original[laplacian == 255] = (0, 0, 255)
        cv2.imwrite(os.path.join(self.out_path_outline , self.fn_original), original)
        return
    
    
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
    Contour(fn_original, src_path, out_path)