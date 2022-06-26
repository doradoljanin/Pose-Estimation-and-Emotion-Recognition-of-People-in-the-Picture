from objectdetection import objectdetection 
from posedetection import *
import cv2
import os.path
import sys
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", "Distutils was imported before Setuptools. This usage is discouraged and may exhibit undesirable behaviors or errors. Please use Setuptools' objects directly or at least import Setuptools first.",  UserWarning, "setuptools.distutils_patch")

input_location = ''
save_location = ''

help_file_location ='help.txt'
# flags

sh_f = False #show help flag
i_l = False #locations
s_l = False

o_d = False #detections
p_d = False
e_d = False

def error_handle(help_file):
    with open(help_file,'r', encoding="utf-8") as f:
            for i,line in enumerate(f):
                if i == 9:
                    print(line, end='')
                    print("Try 'python3 det3ction.py --help' for more information.")
                    
def is_path_valid(path,exist):
    if exist and not os.path.isfile(path):
        return False
    _ , ext = os.path.splitext(path)
    if ext not in ['.jpg','.png','.bmp','.jpeg']:
        return False
    return True

if len(sys.argv) == 1:
    print("Wrong number of arguments.")
    error_handle(help_file_location)
    exit(0)
itr = iter(range(1,len(sys.argv)))
for i in itr:
    current_arg = sys.argv[i]
    if current_arg == '--help':
        sh_f = True
        break
    if current_arg == '--object':
        o_d = True
        continue
    if current_arg == '--pose':
        p_d = True
        continue
    if current_arg == '--emotion':
        e_d = True
        continue
    if current_arg == '--all':
        o_d = True
        p_d = True
        e_d = True
        continue
    if current_arg[0] == '-':
        if 'h' in current_arg:
            sh_f = True
            break
        if 'o' in current_arg:
            s_l = True
            if i < len(sys.argv)-1:
                save_location = sys.argv[i+1]
            else:
                print("Wrong number of arguments.")
                error_handle(help_file_location)
                exit(0)
            next(itr)
        if 'a' in current_arg:
            o_d = True
            p_d = True
            e_d = True
            continue
        if 'd' in current_arg:
            o_d = True
        if 'p' in current_arg:
            p_d = True
        if 'e' in current_arg:
            e_d = True
    elif not i_l:
        input_location = sys.argv[i]
        i_l = True
    else:
        print("Wrong number of arguments.")
        error_handle(help_file_location)
        
        exit(0)

if sh_f:
    with open(help_file_location,'r', encoding="utf-8") as f:
        for line in f:
            print(line, end='')
    print()
    exit(1)
input_location = os.path.abspath(input_location)
save_location = os.path.abspath(save_location)
if not is_path_valid(input_location,True) or not is_path_valid(save_location,False):    
    print("Path not valid!")
    error_handle(help_file_location)
    exit(0)
if not i_l:
    print("No input image.")
    error_handle(help_file_location)
    
    exit(0)


temp = cv2.imread(input_location)
input_path_head,input_path_tail = os.path.split(input_location)

if o_d or p_d or e_d:
    if p_d:
        temp = pose_detection.poseDetection(input_location)
    if o_d:
        rectangles = objectdetection.get_people_coordinates(input_location)
        for i in rectangles:
            cv2.rectangle(temp,(i[0],i[1]),(i[2],i[3]),(0,0,255),2)
    if e_d:
        import emotion_detection
        
        rectangles = emotion_detection.get_emotions(input_location)
        for i in rectangles:
            cv2.rectangle(temp,i[0],i[1],(255,0,0),2)
            text_position = (i[0][0],i[0][1]-10) #left upper corner
            cv2.putText(temp,i[2],text_position, cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
if not s_l:
    save_location = input_path_head + '/a.jpg'
cv2.imwrite(save_location,temp)

