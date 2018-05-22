import io
import sys
import os
import re

def openFile(file_path):
    data = []
    with io.open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()	
            nums = line.split(',')
            if len(nums)>0:
                tmp = []
                for num in nums:
                    tmp.append(float(num))
                	# print(num)
                data.append(tmp)
                #cnt+=1
    print(data[0])            
    return data            	
       
#    return 
def main(feature_file, label_file):
    openFile(feature_file)
    # openFile(label_file)	

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2]) 