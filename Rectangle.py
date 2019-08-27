from __future__ import print_function
import os
import cv2
import argparse
import numpy
import itertools
import time


class Rectangle:
    
    def __init__(self, path_contour, fn_original, src_path, out_path, obj_mag=50, material='C',ftr=542):
        self.obj_mag=obj_mag
        self.fn_original = fn_original
        self.path_contour = path_contour
        self.src_path = src_path
        self.material = material
        self.filter = ftr
        if self.material=='C':
            self.out_path = out_path+'\\PotentialFlakes'
            if ftr>580:
                self.out_path = out_path+'\\PotentialFlakes'
            self.chunks_out_path = out_path+'\\3_ROIs\\'+fn_original[:-4]
            self.out_path_Flakescores = out_path+'\\EstimatedFlakeSize'

        if self.material =='Sb':
            self.out_path = out_path+'\\PotentialFlakes'
            self.chunks_out_path = out_path+'\\3_ROIs\\'+fn_original[:-4]
            self.out_path_Flakescores = out_path+'\\EstimatedFlakeSize'
            
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        if not os.path.exists(self.chunks_out_path):
            os.makedirs(self.chunks_out_path)
        if not os.path.exists(self.out_path_Flakescores):
            os.makedirs(self.out_path_Flakescores)
        self.bounding_box_list = []
        self.ROI_list = []
        self.litho_list = []
        
        # optional
        path_original = os.path.join(self.src_path, self.fn_original)
        original = cv2.imread(path_original,0)
        w, h = original.shape[::-1]
        
        self.min_flake_size = 25
        
        LithoExamplarDir = os.path.join( 'LithoExamplar',str(self.obj_mag))
        
        temp = cv2.imread(os.path.join(LithoExamplarDir ,'cross.png'),0)

        self.coords = []
        self.coords.append(temp)
        
        for i in range(1,10):
            temp = cv2.imread(os.path.join(LithoExamplarDir,(str(i)+'.png')),0)
            self.coords.append(temp)
        self.error_margin = 50
        
    def Lithomatcher(self):
        pass

    
    def markROIs(self, save=True, presentation=1):
        self.ROI_list = []
        path_contour = os.path.join(self.path_contour, self.fn_original)
        path_original = os.path.join(self.src_path, self.fn_original)
        
        
        original = cv2.imread(path_original)
        img_gray = cv2.imread(path_original,0)
        w, h = img_gray.shape[::-1]     # for 10x objective cutting only
        if self.obj_mag==10:
            img_gray = img_gray[:int(h*0.6),:int(w*0.6)]
            original = original[:int(h*0.6),:int(w*0.6)]


        
        mask = numpy.zeros_like(img_gray)
        if self.filter<600:
            for i in range(0,10):
                w_temp, h_temp = self.coords[i].shape[::-1] 
                res = cv2.matchTemplate(img_gray,self.coords[i],cv2.TM_CCOEFF_NORMED)
                threshold = 0.7
                #mask.fill(255)
                loc = numpy.where( res >= threshold)
                matches = []
    
                for pt in zip(*loc[::-1]):
                    distinct_pt = True
                    for match in matches:
                        if abs(pt[0]-match[0])<5*self.obj_mag and abs(pt[1]-match[1])<5*self.obj_mag:
                            distinct_pt = False
                            break
                    if (distinct_pt):
                        mask[pt[1]:int(pt[1]+self.obj_mag/2), pt[0]:int(pt[0]+self.obj_mag/2)] = 255
                        #print(str(i)+' symbol is found at point  '+str(pt))
                        litho_rect = [pt[0], pt[1], int(self.obj_mag/2), int(self.obj_mag/2)]
                        self.litho_list.append(litho_rect)
                        matches.append(pt)
                    
        '''
        coord_cnts, _ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in coord_cnts:
            litho_rect = cv2.minAreaRect(cnt)
            self.litho_list.append(litho_rect)
            print(self.litho_list)
        '''
            
            
        original_for_output= cv2.imread(path_original)
        imgwidth,imgheight = img_gray.shape[::-1]
        
        
        contours = cv2.imread(path_contour,0)
        
        '''
        inv_mask = cv2.bitwise_not(mask)
        contours = cv2.bitwise_and(contours, inv_mask)
        cv2.imshow("contours",contours)
        cv2.waitKey(0) 
        '''
        ret,thresh = cv2.threshold(contours,127,255,cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(contours,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(img_gray, contours, -1, 0, -1)
          
        contours = sorted(contours, key = cv2.contourArea, reverse = True)  # only retain the biggest among overlapping ROIs
        annotations = numpy.empty((0,4), int)
        counter = 0
        
        
        for cnt in contours:
            '''
            single_cnt_mask = numpy.zeros_like(img_gray)
            cv2.drawContours(single_cnt_mask, [cnt], 0, 255, 3)
            if cv2.countNonZero(cv2.bitwise_and(single_cnt_mask, inv_mask)) == 0:
                continue
            '''
            valid= True
            x,y,w,h = cv2.boundingRect(cnt)
   
            
            minArect = cv2.minAreaRect(cnt)
            flake_area_pixels = cv2.contourArea(cnt)-2*cv2.arcLength(cnt,True)
                
            # for checking overlap
            rect = [0,0,0,0]
            rect[0] = max(0,x)
            rect[1] = max(0,y)
            rect[2] = min(int(x+w),imgwidth-1) - rect[0]
            rect[3] = min(int(y+h),imgheight-1) - rect[1]
            rect = tuple(rect)
            
            '''
            for existingrect in self.ROI_list:
                if (self.intersection(existingrect, rect)):
                    valid = False
            '''
            if self.filter<600:
                for existingrect in self.litho_list:
                    if (self.intersection(existingrect, rect)):
                        valid = False  
            
                    
            ''' @TODO
            # overlap guard
            for existingrect in self.ROI_list:
                if (cv2.rotatedRectangleIntersection(existingrect, minArect) != cv2.INTERSECT_NONE ):
                    valid = False
            for existingrect in self.litho_list:
                #print(minArect)
                #print(cv2.boxPoints(minArect))
                if (cv2.rotatedRectangleIntersection(existingrect, minArect) != cv2.INTERSECT_NONE):
                    #print('arrived')
                    valid = False       
            '''
            y_lower = max(0,int(y-h*0.2))
            y_upper = min(int(y+h*1.2),imgheight-1)
            x_lower = max(0,int(x-w*0.2))
            x_upper = min(int(x+w*1.2),imgwidth-1)
            
            if (valid and (y_upper-y_lower > self.min_flake_size)  and (x_upper-x_lower > self.min_flake_size) and (y_upper-y_lower < imgheight)  and (x_upper-x_lower < imgwidth)) :
                roi = original[y_lower:y_upper,x_lower:x_upper]
                counter += 1
                '''self.ROI_list.append(minArect)'''
                self.ROI_list.append(rect)
                bounding_box = cv2.boxPoints(minArect)
                bounding_box = numpy.int0(bounding_box)
                self.bounding_box_list.append(bounding_box)
                cv2.drawContours(original_for_output,[bounding_box],0,(0,255,0),2)
                #cv2.rectangle(original_for_output,(x_lower ,y_lower ),(x_upper,y_upper),(220,220,220),2) 
                areafactor = 1
                if self.obj_mag == 50:
                    areafactor = 73
                if self.obj_mag == 10:
                    areafactor = 2.9
                text_list = numpy.array([bounding_box[1][0], bounding_box[1][1], round(flake_area_pixels/areafactor,2),counter])
                annotations = numpy.concatenate( ( annotations , [text_list] ) , axis=0)
                if save:
                    cv2.imwrite(os.path.join(self.chunks_out_path , (self.fn_original[:-4]+'_'+str(counter).zfill(2)+'.png')), roi)
                    cv2.imwrite(os.path.join(self.out_path , (self.fn_original[:-4]+'_'+str(counter).zfill(2)+'.png')), roi)
                
        #roi = original[y_lower:y_upper,x_lower:x_upper]
        
        #areas = numpy.array([])
            
        '''
        for rectangle in  self.ROI_list :
            x,y,w,h = rectangle
            # slicing for local optical contrast
            y_lower = max(0,int(y-h*0.1))
            y_upper = min(int(y+h+h*0.1),imgheight)
            x_lower = max(0,int(x-w*0.1))
            x_upper = min(int(x+w+w*0.1),imgwidth)
            cv2.rectangle(original_for_output,(x_lower ,y_lower ),(x_upper,y_upper),(200,200,200),4) 
            #rect = cv2.minAreaRect(cnt)
        '''
        
        for ant in annotations:
            text = ('#'+str(int(ant[3]))+'|'+str(ant[2]))
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(original_for_output,text,(int(ant[0]+10),int(ant[1])), font,1,(255,255,255),1,cv2.LINE_AA)  
        
        #cv2.imshow("roi marker",original)
        #cv2.waitKey(0) 

        
        if presentation and not self.material=='Sb':
            win1 = str('Annotated ROIs for '+self.fn_original[:-4])
            cv2.namedWindow(win1)      
            cv2.moveWindow(win1, 0,0)
            cv2.imshow(win1, cv2.resize(original_for_output, (768,768)))
            key = cv2.waitKey(presentation)
            #if self.material=='Sb': time.sleep(2)
            if key == 27: #ESC
                cv2.destroyAllWindows()
            
        cv2.imwrite(os.path.join(self.out_path_Flakescores,self.fn_original ), original_for_output)
        return annotations
    
    
    
    
    def union(self,a,b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return (x, y, w, h)

    def intersection(self,a,b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w<0 or h<0: return False
        return True
    
    
    '''
    def getROIs(self):
        self.ROI_list = []
        path_contour = os.path.join(self.path_contour, self.fn_original)
        path_original = os.path.join(self.src_path, self.fn_original)
        contours = cv2.imread(path_contour)
        original = cv2.imread(path_original)
        imgheight, imgwidth, _ = original.shape
        contours = cv2.cvtColor(contours, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(contours,127,255,cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(contours,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        counter = 0 
        valid = True
        for cnt in contours:
            valid= True
            x,y,w,h = cv2.boundingRect(cnt)
            rect = [0,0,0,0]
            y_lower = max(0,int(y-h*0.5))
            y_upper = min(int(y+h+h*0.5),imgheight)
            x_lower = max(0,int(x-w*0.5))
            x_upper = min(int(x+w+w*0.5),imgwidth)
            rect[0] = max(0,x)
            rect[1] = max(0,y)
            rect[2] = min(int(x+w),imgwidth) - rect[0]
            rect[3] = min(int(y+h),imgheight) - rect[1]
            rect = tuple(rect)
            
            for existingrect in self.ROI_list:
                if (self.intersection(existingrect, rect)):
                    valid = False

            roi = original[y_lower:y_upper,x_lower:x_upper]
            if (valid and (y_upper-y_lower > self.min_flake_size)  and (x_upper-x_lower > self.min_flake_size) and (y_upper-y_lower < imgheight)  and (x_upper-x_lower < imgwidth)) :
                self.ROI_list.append(rect)
                counter += 1
                rect = cv2.minAreaRect(cnt)
                bounding_box = cv2.boxPoints(rect)
                bounding_box = numpy.int0(bounding_box)
                self.bounding_box_list.append(bounding_box) 
                cv2.imwrite(os.path.join(self.chunks_out_path , (self.fn_original[:-4]+'_'+str(counter)+'.png')), roi)

                        
        #cv2.imwrite(os.path.join(self.out_path , self.fn_original), original)
        return self.bounding_box_list
    '''
        
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('path_contour',type=str)
        parser.add_argument('fn_original',type=str)
        parser.add_argument('src_path',type=str)
        parser.add_argument('out_path',type=str)
        
        args = parser.parse_args()
        
        path_contour = args.path_contour
        fn_original = args.fn_original
        src_path = args.src_path
        out_path = args.out_path
        
    except:
        pass
    Rectangle(path_contour, fn_original, src_path, out_path)