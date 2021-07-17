# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 15:49:30 2021

@author: tales
"""

import math 
import face_recognition


def dist(x0, y0, x1, y1):
    a = (x1-x0)**2 + (y1-y0)**2
    b = math.sqrt(a)
    return b

def add(a, lista):
    lista.append(a)
    return print("Sucesso")

    
def calculoAU (frameNeutro, frameApice):
    #guarda os landmarks em um dicionário
    face_landmarks_list_neutro = face_recognition.face_landmarks(frameNeutro)
    
    face_landmarks_list_apice = face_recognition.face_landmarks(frameApice)
    
    #Inner Brow Raiser
    AU1Variation = dist(face_landmarks_list_apice[0]['left_eyebrow'][4][0], face_landmarks_list_apice[0]['left_eyebrow'][4][1],
                        face_landmarks_list_apice[0]['left_eye'][3][0], face_landmarks_list_apice[0]['left_eye'][3][1]) - dist(face_landmarks_list_neutro[0]['left_eyebrow'][4][0], face_landmarks_list_neutro[0]['left_eyebrow'][4][1],
                        face_landmarks_list_neutro[0]['left_eye'][3][0], face_landmarks_list_neutro[0]['left_eye'][3][1]) 
    #Inner Brow Raiser
    AU11Variation = dist(face_landmarks_list_apice[0]['right_eyebrow'][0][0], face_landmarks_list_apice[0]['right_eyebrow'][0][1],
                         face_landmarks_list_apice[0]['right_eye'][0][0], face_landmarks_list_apice[0]['right_eye'][0][1]) - dist(face_landmarks_list_neutro[0]['right_eyebrow'][0][0], face_landmarks_list_neutro[0]['right_eyebrow'][0][1],
                         face_landmarks_list_neutro[0]['right_eye'][0][0], face_landmarks_list_neutro[0]['right_eye'][0][1])
    #Outer Brow Raiser
    AU21Variation = dist(face_landmarks_list_apice[0]['left_eyebrow'][1][0], face_landmarks_list_apice[0]['left_eyebrow'][1][1],
                         face_landmarks_list_apice[0]['left_eye'][3][0], face_landmarks_list_apice[0]['left_eye'][3][1]) - dist(face_landmarks_list_neutro[0]['left_eyebrow'][1][0], face_landmarks_list_neutro[0]['left_eyebrow'][1][1],
                         face_landmarks_list_neutro[0]['left_eye'][3][0], face_landmarks_list_neutro[0]['left_eye'][3][1])
    
    #Outer Brow Raiser
    AU22Variation = dist(face_landmarks_list_apice[0]['right_eyebrow'][3][0], face_landmarks_list_apice[0]['right_eyebrow'][3][1],
                         face_landmarks_list_apice[0]['right_eye'][0][0], face_landmarks_list_apice[0]['right_eye'][0][1]) - dist(face_landmarks_list_neutro[0]['right_eyebrow'][3][0], face_landmarks_list_neutro[0]['right_eyebrow'][3][1],
                         face_landmarks_list_neutro[0]['right_eye'][0][0], face_landmarks_list_neutro[0]['right_eye'][0][1])
    
    #Brow Lowerer
    AU4Variation = dist(face_landmarks_list_apice[0]['left_eyebrow'][4][0], face_landmarks_list_apice[0]['left_eyebrow'][4][1],
                        face_landmarks_list_apice[0]['right_eyebrow'][0][0], face_landmarks_list_apice[0]['right_eyebrow'][0][1]) - dist(face_landmarks_list_neutro[0]['left_eyebrow'][4][0], face_landmarks_list_neutro[0]['left_eyebrow'][4][1],
                        face_landmarks_list_neutro[0]['right_eyebrow'][0][0], face_landmarks_list_neutro[0]['right_eyebrow'][0][1])
    
    #Upper Lid Raiser
    AU51Variation = dist(face_landmarks_list_apice[0]['left_eye'][2][0], face_landmarks_list_apice[0]['left_eye'][2][1],
                         face_landmarks_list_apice[0]['left_eye'][4][0], face_landmarks_list_apice[0]['left_eye'][4][1]) - dist(face_landmarks_list_neutro[0]['left_eye'][2][0], face_landmarks_list_neutro[0]['left_eye'][2][1],
                         face_landmarks_list_neutro[0]['left_eye'][4][0], face_landmarks_list_neutro[0]['left_eye'][4][1])
    
    #Upper Lid Raiser
    AU52Variation = dist(face_landmarks_list_apice[0]['right_eye'][1][0], face_landmarks_list_apice[0]['right_eye'][1][1],
                         face_landmarks_list_apice[0]['right_eye'][5][0], face_landmarks_list_apice[0]['left_eye'][5][1]) - dist(face_landmarks_list_neutro[0]['right_eye'][1][0], face_landmarks_list_neutro[0]['right_eye'][1][1],
                         face_landmarks_list_neutro[0]['right_eye'][5][0], face_landmarks_list_neutro[0]['left_eye'][5][1])
    
    #Cheek Raiser
    AU61Variation = AU51Variation
    
    #Cheek Raiser
    AU62Variation = AU52Variation
    
    #Nose Wrinkler
    AU91Variation = dist(face_landmarks_list_apice[0]['left_eyebrow'][4][0], face_landmarks_list_apice[0]['left_eyebrow'][4][1],
                         face_landmarks_list_apice[0]['nose_tip'][0][0], face_landmarks_list_apice[0]['nose_tip'][0][1]) - dist(face_landmarks_list_neutro[0]['left_eyebrow'][4][0], face_landmarks_list_neutro[0]['left_eyebrow'][4][1],
                         face_landmarks_list_neutro[0]['nose_tip'][0][0], face_landmarks_list_neutro[0]['nose_tip'][0][1])
    
    #Nose Wrinkler
    AU92Variation = dist(face_landmarks_list_apice[0]['right_eyebrow'][0][0], face_landmarks_list_apice[0]['right_eyebrow'][0][1],
                         face_landmarks_list_apice[0]['nose_tip'][4][0], face_landmarks_list_apice[0]['nose_tip'][4][1]) - dist(face_landmarks_list_neutro[0]['right_eyebrow'][0][0], face_landmarks_list_neutro[0]['right_eyebrow'][0][1],
                         face_landmarks_list_neutro[0]['nose_tip'][4][0], face_landmarks_list_neutro[0]['nose_tip'][4][1])
    
    
    #Upper Lip Raiser
    AU10Variation = dist(face_landmarks_list_apice[0]['top_lip'][3][0], face_landmarks_list_apice[0]['top_lip'][3][1],
                         face_landmarks_list_apice[0]['nose_tip'][2][0], face_landmarks_list_apice[0]['nose_tip'][2][1]) - dist(face_landmarks_list_neutro[0]['top_lip'][3][0], face_landmarks_list_neutro[0]['top_lip'][3][1],
                         face_landmarks_list_neutro[0]['nose_tip'][2][0], face_landmarks_list_neutro[0]['nose_tip'][2][1])
    
    #Lip Corner Puller
    AU12Variation = dist(face_landmarks_list_apice[0]['top_lip'][0][0], face_landmarks_list_apice[0]['top_lip'][0][1],
                         face_landmarks_list_apice[0]['top_lip'][6][0], face_landmarks_list_apice[0]['top_lip'][6][1]) - dist(face_landmarks_list_neutro[0]['top_lip'][0][0], face_landmarks_list_neutro[0]['top_lip'][0][1],
                         face_landmarks_list_neutro[0]['top_lip'][6][0], face_landmarks_list_neutro[0]['top_lip'][6][1])
    
    #Lip Corner Depressor
    AU151Variation = dist(face_landmarks_list_apice[0]['nose_tip'][0][0], face_landmarks_list_apice[0]['nose_tip'][0][1],
                          face_landmarks_list_apice[0]['top_lip'][0][0], face_landmarks_list_apice[0]['top_lip'][0][1]) - dist(face_landmarks_list_neutro[0]['nose_tip'][0][0], face_landmarks_list_neutro[0]['nose_tip'][0][1],
                          face_landmarks_list_neutro[0]['top_lip'][0][0], face_landmarks_list_neutro[0]['top_lip'][0][1])
    
    #Lip Corner Depressor
    AU152Variation = dist(face_landmarks_list_apice[0]['nose_tip'][4][0], face_landmarks_list_apice[0]['nose_tip'][4][1],
                          face_landmarks_list_apice[0]['top_lip'][6][0], face_landmarks_list_apice[0]['top_lip'][6][1]) - dist(face_landmarks_list_neutro[0]['nose_tip'][4][0], face_landmarks_list_neutro[0]['nose_tip'][4][1],
                          face_landmarks_list_neutro[0]['top_lip'][6][0], face_landmarks_list_neutro[0]['top_lip'][6][1])
    
    #Chin Raiser
    AU17Variation = dist(face_landmarks_list_apice[0]['bottom_lip'][3][0], face_landmarks_list_apice[0]['bottom_lip'][3][1],
                         face_landmarks_list_apice[0]['nose_tip'][2][0], face_landmarks_list_apice[0]['nose_tip'][2][1]) - dist(face_landmarks_list_neutro[0]['bottom_lip'][3][0], face_landmarks_list_neutro[0]['bottom_lip'][3][1],
                         face_landmarks_list_neutro[0]['nose_tip'][2][0], face_landmarks_list_neutro[0]['nose_tip'][2][1])
    
    #Lip stretcher
    AU20Variation = dist(face_landmarks_list_apice[0]['bottom_lip'][3][0], face_landmarks_list_apice[0]['bottom_lip'][3][1],
                         face_landmarks_list_apice[0]['chin'][8][0], face_landmarks_list_apice[0]['chin'][8][1]) - dist(face_landmarks_list_neutro[0]['bottom_lip'][3][0], face_landmarks_list_neutro[0]['bottom_lip'][3][1],
                         face_landmarks_list_neutro[0]['chin'][8][0], face_landmarks_list_neutro[0]['chin'][8][1])
    
    #Lip Tightener
    AU23Variation = dist(face_landmarks_list_apice[0]['top_lip'][0][0], face_landmarks_list_apice[0]['top_lip'][0][1],
                         face_landmarks_list_apice[0]['top_lip'][6][0], face_landmarks_list_apice[0]['top_lip'][6][1]) - dist(face_landmarks_list_neutro[0]['top_lip'][0][0], face_landmarks_list_neutro[0]['top_lip'][0][1],
                         face_landmarks_list_neutro[0]['top_lip'][6][0], face_landmarks_list_neutro[0]['top_lip'][6][1])
    
    #Lip Pressor
    AU24Variation = dist(face_landmarks_list_apice[0]['top_lip'][3][0], face_landmarks_list_apice[0]['top_lip'][3][1],
                         face_landmarks_list_apice[0]['bottom_lip'][3][0], face_landmarks_list_apice[0]['bottom_lip'][3][1]) - dist(face_landmarks_list_neutro[0]['top_lip'][3][0], face_landmarks_list_neutro[0]['top_lip'][3][1],
                         face_landmarks_list_neutro[0]['bottom_lip'][3][0], face_landmarks_list_neutro[0]['bottom_lip'][3][1])
    
    #Lips Part
    AU25Variation = dist(face_landmarks_list_apice[0]['top_lip'][9][0], face_landmarks_list_apice[0]['top_lip'][9][1],
                         face_landmarks_list_apice[0]['bottom_lip'][9][0], face_landmarks_list_apice[0]['bottom_lip'][9][1]) - dist(face_landmarks_list_neutro[0]['top_lip'][9][0], face_landmarks_list_neutro[0]['top_lip'][9][1],
                         face_landmarks_list_neutro[0]['bottom_lip'][9][0], face_landmarks_list_neutro[0]['bottom_lip'][9][1])
    
    #Jaw Drop
    AU26Variation = dist(face_landmarks_list_apice[0]['chin'][8][0], face_landmarks_list_apice[0]['chin'][8][1],
                         face_landmarks_list_apice[0]['nose_tip'][2][0], face_landmarks_list_apice[0]['nose_tip'][2][1]) - dist(face_landmarks_list_neutro[0]['chin'][8][0], face_landmarks_list_neutro[0]['chin'][8][1],
                         face_landmarks_list_neutro[0]['nose_tip'][2][0], face_landmarks_list_neutro[0]['nose_tip'][2][1])
    
    #Mouth Stretch
    AU27Variation = AU26Variation 
    
    
    return AU1Variation, AU11Variation, AU21Variation, AU22Variation, AU4Variation, AU51Variation, AU52Variation, AU61Variation, AU62Variation, AU91Variation, AU92Variation, AU10Variation, AU12Variation, AU151Variation, AU152Variation, AU17Variation, AU20Variation, AU23Variation, AU24Variation, AU25Variation, AU26Variation, AU27Variation                                 


def EmotionMasculino(lista):
    
    if (lista[0] >= 1.89 and lista[0] <= 13.39) and (lista[1] >= 3.13 and lista[1] <= 12.42) and (lista[13] >= 1.46 and lista[13] <= 23.24) and (lista[14] >= 0.37 and lista[14] <= 29.99) and (lista[15] >= -10 and lista[15] <= 8.97):
        #print("sadness")
        #A1, A15, A17
        return  6
    
    if (lista[4] > -16.9 and lista[4] <= 0) and (lista[18] >= -18 and lista[18] < -1.90) and (lista[20] >= -16.92 and lista[20] <= 20.97):
        #print("raiva")
        #AU4, A24, AU26
        return  1
    
    if (lista[9] >= -20.42 and lista[9] <= -7.82) and (lista[10] >= -19.3 and lista[10] <= -9.10) and (lista[11] >= -13.64 and lista[11] <= -0.05) and (lista[19] >= 0 and lista[19] <= 14.03):
        #print("nojo")
        #AU10, AU25, AU9
        return  3
    
    if (lista[0] >= 4.39 and lista[0] <= 13.73) and (lista[1] >= 3.56 and lista[1] <= 15.77) and (lista[20] >= 0.94 and lista[20] <= 11.99):
        #print("fear")
        #AU1, AU26
        return  4
    
    if (lista[8] >= -7.03 and lista[8] < -1.13) and (lista[12] >= 12.86 and lista[12] < 36):
        #print("Alegria")
        #AU6, AU12
        return  5

    if (lista[20] >= 19 and lista[20] <= 56.94) and (lista[5] >= 0.09 and lista[5] <= 9.02) and (lista[6] >= 0. and lista[6] <= 8.99):
        #print("surpresa")
        #AU26, AU5
        return  7

def EmotionFeminino(lista):
    
    if (lista[0] >= 1.89 and lista[0] <= 13.39) and (lista[1] >= 3.13 and lista[1] <= 12.42) and (lista[13] >= 1.46 and lista[13] <= 23.24) and (lista[14] >= 0.37 and lista[14] <= 29.99) and (lista[15] >= -10 and lista[15] <= 8.97):
        #print("sadness")
        #A1, A15, A17
        return  6
    
    if (lista[4] > -16.9 and lista[4] <= 0) and (lista[18] >= -18 and lista[18] < -1.90) and (lista[20] >= -16.92 and lista[20] <= 20.97):
        #print("raiva")
        #AU4, A24, AU26
        return  1
    
    if (lista[9] >= -20.42 and lista[9] <= -7.82) and (lista[10] >= -19.3 and lista[10] <= -9.10) and (lista[11] >= -13.64 and lista[11] <= -0.05) and (lista[19] >= 0 and lista[19] <= 14.03):
        #print("nojo")
        #AU10, AU25, AU9
        return  3
    
    if (lista[0] >= 4.39 and lista[0] <= 13.73) and (lista[1] >= 3.56 and lista[1] <= 15.77) and (lista[20] >= 0.94 and lista[20] <= 11.99):
        #print("fear")
        #AU1, AU26
        return  4
    
    if (lista[8] >= -7.03 and lista[8] < -1.13) and (lista[12] >= 12.86 and lista[12] < 36):
        #print("Alegria")
        #AU6, AU12
        return  5

    if (lista[20] >= 19 and lista[20] <= 56.94) and (lista[5] >= 0.09 and lista[5] <= 9.02) and (lista[6] >= 0. and lista[6] <= 8.99):
        #print("surpresa")
        #AU26, AU5
        return  7
    
    



def Anger(lista):
    if (lista[4] > -16.9 and lista[4] <= 0) and (lista[18] >= -18 and lista[18] < -1.90) and (lista[20] >= -16.92 and lista[20] <= 20.97):
        print("raiva")
        return "raiva"

def Hapiness(lista):
    if (lista[8] >= -7.03 and lista[8] < -1.13) and (lista[12] >= 12.86 and lista[12] < 36):
        print("Alegria")
        return "Alegria"

def Surprise(lista):
    if (lista[20] >= 19 and lista[20] <= 56.94) and (lista[5] >= 0.09 and lista[5] <= 9.02) and (lista[6] >= 0. and lista[6] <= 8.99):
        print("surpresa")
        return "surpresa"
    
def Sadness(lista):
    if (lista[0] >= 1.89 and lista[0] <= 13.39) and (lista[1] >= 3.13 and lista[1] <= 12.42) and (lista[13] >= 1.46 and lista[13] <= 23.24) and (lista[14] >= 0.37 and lista[14] <= 29.99) and (lista[15] >= -10 and lista[15] <= 8.97):
        print("sadness")
        return "sadness"

def Fear(lista):
    if (lista[0] >= 4.39 and lista[0] <= 13.73) and (lista[1] >= 3.56 and lista[1] <= 15.77) and (lista[20] >= 0.94 and lista[20] <= 11.99):
        return "fear"

def Disgust(lista):
    if (lista[9] >= -20.42 and lista[9] <= -7.82) and (lista[10] >= -19.3 and lista[10] <= -9.10) and (lista[11] >= -13.64 and lista[11] <= -0.05) and (lista[19] >= 0 and lista[19] <= 14.03):
        print("nojo")
        return "nojo"
    
    















    
