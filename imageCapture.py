# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:14:24 2021

@author: tales
"""

import cv2
from functions import calculoAU, EmotionMasculino, EmotionFeminino
import face_recognition
import csv

s = 0 

i = 0.
j = 0.

csvHeader = ['ID', 'FolderName', 'EmotionNumber', 'Result']

f = open('ExpectedResults.csv', 'w', newline='', encoding='utf-8')

with open('ExpectedResults.csv', 'w', newline='') as file:
    
    writer = csv.writer(file)
    
    writer.writerow(csvHeader)

f.close() 

while(True):
    
    nome = input(print("Digite o nome do arquivo csv:"))
    
    f = open(nome + '.csv', 'r')
    
    try:
        reader = csv.reader(f, delimiter = ";")
        for line in reader:
            #line[0] -> ID, line[1] -> gender, line[2] -> folderName, line[3] -> emotionNumber
            #line[4l-> emotionName, line[5] -> AUs detectadas de acordo com o banco, 
            #line[6] ->file name neutra, line[7] ->file name ápice, line[8]-> não classificado
            if s > 1:
                
                if line[2] == '10' or line[2] == '11' or line[2] == '12' or line[2] == '13':
            
                    frameNeutro = cv2.imread("/Users/tales/Desktop/Projeto_PIBIC/extended-cohn-kanade-images/cohn-kanade-images" + "/" + str(line[0])+ "/0"+ str(line[2])+ "/"+ str(line[6]) + ".png")
            
                    frameApice = cv2.imread("/Users/tales/Desktop/Projeto_PIBIC/extended-cohn-kanade-images/cohn-kanade-images" + "/" + str(line[0])+ "/0"+ str(line[2])+ "/"+ str(line[7]) + ".png")
                else:
                    frameNeutro = cv2.imread("/Users/tales/Desktop/Projeto_PIBIC/extended-cohn-kanade-images/cohn-kanade-images" + "/" + str(line[0])+ "/00"+ str(line[2])+ "/"+ str(line[6]) + ".png")
            
                    frameApice = cv2.imread("/Users/tales/Desktop/Projeto_PIBIC/extended-cohn-kanade-images/cohn-kanade-images" + "/" + str(line[0])+ "/00"+ str(line[2])+ "/"+ str(line[7]) + ".png")
            
                
                if (not line[8] == "X") and (not line[2] == "2") :
                    result = calculoAU(frameNeutro, frameApice)
                    
                    if not(EmotionFeminino(result) == None):
                        print(line[0], line[2], line[3], EmotionMasculino(result))
                        
                        l = open('ExpectedResults.csv', 'a', newline = '')
                        try:
                            writer = csv.writer(l, delimiter = ";")
                            writer.writerow((line[0], line[2], line[3], EmotionMasculino(result)))
                        finally:
                            l.close()
                        
                        if line[3] == str(EmotionMasculino(result)):
                            i = i + 1
                        j = j + 1  
                        
                    
                    
        
            
            s = s + 1 
            
    finally:
        f.close()
    break

print(i/j*100, '%')
















