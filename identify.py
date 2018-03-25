#!/usr/bin/env python

# EXTERNAL DEPENDENCIES
import cv2
import numpy as np
from keras.models import load_model
import Levenshtein

import sys
import unicodedata
import os
import json
import time
import copy as cp


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    try:
        text = unicode(input, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)




def parseNameList(namefile):
    if namefile == "":
        raise ValueError("No s'ha introduit cap fitxer csv/txt per a ser utilitzat com a llista de fitxers")


    expectedStrings = []
    data = []
    with open(namefile) as f:
        next(f)
        for idx, line in enumerate(f):
            fields=line.split(";")
            nameSurname = fields[1].split(",")

            surname = nameSurname[0]
            ascii_surname = remove_diacritic(surname.replace(" ", ""))

            name = nameSurname[1]
            ascii_name = remove_diacritic(name.replace(" ", ""))

            dni = fields[2].upper()
            expectedStrings.append(dni+ascii_name.upper()+ascii_surname.upper())
            data.append((dni, name, surname, ascii_name, ascii_surname, idx))


    return data, expectedStrings
    



def parseArgs(argList, prefix):
    images = []
    nameList = ""
    for file in argList:
        file = os.path.join(prefix, file)
        if file.endswith(".jpg"): 
            images.append(file)
        elif os.path.isdir(file):
            i, nl = parseArgs(os.listdir(file), file)
            images.extend(i)
            if nl != "":
                nameList = nl
        elif file.endswith(".txt") or file.endswith(".csv"):
            nameList = file
    return (images, nameList)

def main():

    modelNIST = load_model("./checkpoint_nonums.h5")
    modelMNIST = load_model("./checkpoint_mnist.h5")
    models = (modelNIST, modelMNIST)
    modelMNIST.summary()
    modelNIST.summary()

    images, nameList = parseArgs(sys.argv, "")
    names = parseNameList(nameList)

    results = []
    errors = []
    for image in images:
        fileResult, errorRates = processFile(image, names[1], models)
        results.append(fileResult)
        errors.append(errorRates)

    result,unused, unrecognizable = mapResults(results, names[0], images, errors)
    fin = {
        "identified" : result, 
        "unused" : unused, 
        "unrecognizable" : unrecognizable
    }

    json.dump(fin, sys.stdout)

def resizeImage(img, width):
    oHeight, oWidth = img.shape[:2]

    height = int((width/float(oWidth)) * oHeight)
    return cv2.resize(img, (width,height))


def binarize(img, threshold, numberOfBlurs):
    tmp = img
    for i in range(numberOfBlurs):
        tmp = cv2.blur(tmp, (3,3)) 

    _, tmp = cv2.threshold(tmp, threshold, 255, cv2.THRESH_BINARY)
    return tmp


def sortContours(c):
    x,y,w,h = cv2.boundingRect(c)
    return w 


def findShapes(img, minimumArea, expectedHeight, topPortion):
    if topPortion != None:
        h, w = img.shape[:2]
        nh = int(round(h*topPortion))
        img = img[0:nh, 0:w]

    r = []
    _, cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print ("hola")
    print (cnts[0])
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w * h > minimumArea and h > expectedHeight * 0.7 and h < expectedHeight * 1.3 :
            r.append(c)
    return r

def test(im):

    _, cnts, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, cnts2, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Copy over the original image to separate variables
    img1 = im.copy()
    img2 = im.copy()

    # Draw both contours onto the separate images
    cv2.drawContours(img1, cnts, -1, (0,0,0), 3)
    cv2.drawContours(img2, cnts2, -1, (0,0,0), 3)

    out = np.hstack([img1, img2])

    # Now show the image
    cv2.imshow('Output', out)
    time.sleep(5)

def predictCharacter(img, boundingBox, model) :
    x, y, w, h = boundingBox
    portion = img[y:y+h, x:x+w]
    portion = cv2.resize(portion, (64,64))
    arr = np.asarray(portion)
    arr = np.expand_dims(arr, axis=0)
    arr = np.expand_dims(arr, axis=3)
    return model.predict(arr)[0]


def lookForCharactersInImage(image, contour):
    mask = np.zeros((image.shape),np.uint8)
    cv2.fillConvexPoly(mask, contour, 255)
    masked = cv2.bitwise_and(mask, image)
    #cv2.imshow(str(random.random()), masked)
    characterShapes = findShapes(masked, 1000, 60, None)

    sumw = 0
    for i,s in enumerate(characterShapes):
        bb = cv2.boundingRect(s)
        x, y, w, h = bb
        sumw += w

    avgW = sumw/len(characterShapes) if len(characterShapes) > 0 else 0
    
    boundingBoxes = []
    for s in characterShapes:
        bb = cv2.boundingRect(s)
        x, y, w, h = bb
        if w > 1.5 * avgW:
            bb1 = (x,y,w/2, h)
            bb2 = (x + w/2,y,w/2, h)
            boundingBoxes.append(bb1)
            boundingBoxes.append(bb2)
        else:
            boundingBoxes.append(bb)

    boundingBoxes.sort(key=lambda bb: bb[0])
    return boundingBoxes
            



def detectCharacters(image, gray,boundingBoxes, model):
    preds = []
    for i,bb in enumerate(boundingBoxes):
        x, y, w, h = bb
        character = image[y:y+h, x:x+w]
        channels = cv2.split(character)
        hist = cv2.calcHist(channels,[1], None, [1], [0,250])
        if hist[0] > 300:
            predVector = predictCharacter(gray, bb, model)
            preds.append(predVector)

    return preds

def predictWithBestCharGuess(vectorPredictions, startingChar, skipLast):
    ret = ""
    lastIndex = len(vectorPredictions) - skipLast  
    for pred in vectorPredictions[:lastIndex]:
        index = np.argmax(pred)
        ret= ret + chr(ord(startingChar) + index)
    return ret


def mapResults(results, namesData, filenames, errorRates):
    usedNames = []
    returnData = []
    unrecognizable = []
    for i in range(len(results)):
        minimumIndex = np.argmin(results)
        filenameIndex = minimumIndex / len(namesData)
        nameIndex = minimumIndex % len(namesData)

        

        data = namesData[nameIndex]
        filename = filenames[filenameIndex]
        errorRate = errorRates[filenameIndex][nameIndex]
        #(dni, name, surname, ascii_name, ascii_surname, idx)
        entry = {
            "dni": data[0],
            "name": data[1],
            "surname": data[2],
            "ascii_name": data[3],
            "ascii_surname": data[4],
            "csv_index": data[5],
            "filename": filename,
            "error": errorRate
        }

        if errorRate != 1.0:
            usedNames.append(nameIndex)
            returnData.append(entry)
            results[filenameIndex] = [9999 for i in range(len(namesData))]
            for result in results:
                result[nameIndex] = 9999

    for i in range(len(results)):
        if results[i][0] != 9999 :
            unrecognizable.append(filenames[i])

    for idx in usedNames:
        namesData[idx] = None
    
    unusedEntries = []
    for data in namesData:
        if data != None:
            unusedEntry = {
                "dni" : data[0],
                "name" : data[1],
                "surname" : data[2],
                "ascii_name" : data[3],
                "ascii_surname": data[4],
                "csv_index": data[5]
            }
            unusedEntries.append(unusedEntry)

    return returnData, unusedEntries, unrecognizable

    



def processFile(filename, possibleWords, models):
    
    #print "Processing", filename
    img = cv2.imread(filename)

    #Resize image
    img = resizeImage(img, 1700)

    orig = img 

    h, w = img.shape[:2]



    #Binarize Image
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, gray = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)


    binarized = binarize(img, 248,1)
    img = cv2.bitwise_not(binarized)
    #cv2.imshow("img to find shapes", img)
    #Find Contours and mask everything that is not in the expected contour
    #Daqui surten els contorns en format de vector amb les linies


    #TEST
    imgAux = cp.deepcopy(img[0:int(round(h*0.2)), 0:w])
    test(imgAux)
    textFieldShapes = findShapes(img, 4000, 60, 0.2)
    textFieldShapes.sort(key=sortContours)
    
    totalTextPrediction = ""
    for idx, s in enumerate(textFieldShapes[-3:]):
        boundingBoxes = lookForCharactersInImage(binarized, s)
        #print "Found", len(boundingBoxes), "character shapes"
        
        model = models[1] if idx == 0 else models[0]
        predictions = detectCharacters(orig,gray, boundingBoxes, model)
        textPrediction = predictWithBestCharGuess(predictions, '0', 1) if idx == 0 else predictWithBestCharGuess(predictions, 'A', 0) 
        totalTextPrediction = totalTextPrediction + textPrediction
 

    distances = []
    errorRates = []
    totalTextPredictionLen = len(totalTextPrediction)
    for possibleWord in possibleWords:
        dist = Levenshtein.distance(possibleWord, totalTextPrediction)
        distances.append(dist)
        errorRates.append(dist / float(max(totalTextPredictionLen, len(possibleWord))) )

    cv2.waitKey(5000)
    
    return (distances, errorRates)



if __name__ == "__main__":
    main()