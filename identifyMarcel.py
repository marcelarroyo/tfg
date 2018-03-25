#!/usr/bin/env python

# EXTERNAL DEPENDENCIES
import cv2
import numpy as np
from keras.models import load_model
import Levenshtein
from keras.models import model_from_json
import sys
import unicodedata
import os
import json
import copy as cp
import time
#Input: Directori on buscar el csv i els arxius jpg. /test/test

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
	#print(str(file))
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
    #json_file = open('model.json','r')
    #loaded_model_json = json_file.read()
    #modelMNIST = model_from_json(loaded_model_json)
    #modelMNIST.load_weights("modelMarcel.h5")
    modelMNIST = load_model("./checkpoint_mnist.h5")
    modelNIST = load_model("./checkpoint_nonums.h5")
    models = (modelNIST, modelMNIST)
    modelMNIST.summary()
    modelNIST.summary()

    images, nameList = parseArgs(sys.argv, "")
    names = parseNameList(nameList)

    results = []
    errors = []
    #print "ENTRO AL BUCLE"
    #print images
    for image in images:
	#print "HE ENTRAT AL BUCLE"
        print (names[1])
        print (str(image))
        time.sleep(3)
        #fileResult, errorRates = processFile(image, names[1], models)
        processFile(image, names[1], models)
        #results.append(fileResult)
        #errors.append(errorRates)
'''
    result,unused, unrecognizable = mapResults(results, names[0], images, errors)
    fin = {
        "identified" : result, 
        "unused" : unused, 
        "unrecognizable" : unrecognizable
    }

    json.dump(fin, sys.stdout)
'''
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


def cutImage(img, portion):
    h, w = img.shape[:2]
    inici = int(round(h*0.6))
    imageAuxiliar = cp.deepcopy(img[inici:h, 0:w])
    return imageAuxiliar

def findShapes(img, minimumArea, expectedHeight, topPortion):
    if topPortion != None:
        h, w = img.shape[:2]
        nh = int(round(h*topPortion))
        inici = int(round(h*0.6))
        print(h)
        print(w)
        print(inici)
        print(nh)
        ##imageAuxiliar = cp.deepcopy(img[inici:nh, 0:w])
        img = img[0:nh, 0:w]
    #v2.imshow("contornsRetallats", imat)
    #time.sleep(2

    print(img)
    r = []
    #findContours modifica la imatge tractada. Una vegada troba el primer contorn. Implica fer 3 copies una per cada graella a tractar
    _, cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(cnts)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        #Comprovar si els contorns extrets coincideixen amb els valors esperats
        if w * h > minimumArea and h > expectedHeight * 0.7 and h < expectedHeight * 1.3 :
            r.append(c)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.imshow("contorns" + str(i),img[x:x+w, y:y+h])
        #time.sleep(2)
    return r

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

    

def cutImage(image, minHeight, maxHeight, minWide, maxWide):
    return image[minHeight:maxHeight, minWide:maxWide]

def processFile(filename, possibleWords, models):
    
    #print "Processing", filename
    img = cv2.imread(filename,0)
    #Resize image
    img2 = resizeImage(img, 1700)
    #Suavitza la imatge(difumina)
    blur = cv2.GaussianBlur(img2,(5,5),0)
    #Nega la imatge
    imgBitwise = cv2.bitwise_not(blur)
    #cv2.imshow("HOLA", imgBitwise)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()
    #ret3,th3 = cv2.threshold(imgBitwise,0,255,20)
    #Retalla la imatge per la zona on s'ha d'extreure les taules
    imgBitwise = cutImage(imgBitwise, 120,380,0,1700)
    horizontal = cp.deepcopy(imgBitwise)
    vertical = cp.deepcopy(imgBitwise)
    horizontalsize, verticalsize = imgBitwise.shape[:2]
    horizontalsize = 200
    verticalsize = 55
    #Busca rectes horitzontals amb una distancia major a 200 pixels
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1));
    source_eroded = cv2.erode(horizontal, horizontalStructure, iterations=1);
    source_dilated = cv2.dilate(source_eroded,horizontalStructure, iterations=1);
    #cv2.imshow("horizontal", source_dilated)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()
    #Busca rectes verticals amb una distancia major a 50 pixels
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize));
    source_eroded2 = cv2.erode(vertical, verticalStructure, iterations=1);
    source_dilated2 = cv2.dilate(source_eroded2,verticalStructure, iterations=1);
    #cv2.imshow("vertical", source_dilated2)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()
    #Imprimim les rectes horitzontals i verticals
    table = source_dilated2 + source_dilated
    #cv2.imshow("Table", table)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()
    #Extreiem els vertexs de la taula
    vertexs = cv2.bitwise_and(source_dilated, source_dilated2)
    #cv2.imshow("Vertexs", vertexs)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()
    #Tractar de eliminar la casella de nota
    yCoor = 0
    rowsV = vertexs.shape[0]
    colsV = vertexs.shape[1]
    for x in range (0, rowsV):
        for y in range(0, colsV):
            if (vertexs[x][y] > 20 and y > yCoor):
                yCoor = y
    cleanTable = cp.deepcopy(table)
    rowsT = table.shape[0]
    colsT = table.shape[1]
    #Marge on comencar a pintar la imatge de negre
    marge = yCoor + 10
    for x in range (0, rowsT):
        for y in range(marge, colsT):
            cleanTable[x][y] = 0
    #cv2.imshow("Clean Table", cleanTable)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()
    #Tractar de trobar els contorns de la taula
    '''
    _, contours, hierarchy = cv2.findContours(table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    print(contours)
    for element in contours:
        area = cv2.contourArea(element)
        print(area)
        if (area > 55):
            contour_poly = cv2.approxPolyDP(element, 3, True)
            print(contour_poly)
            boundRect = cv2.boundingRect(contour_poly)
            print(boundRect)
            roi = vertexs[boundRect]
            joints_contour = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if (joints_contour.size() > 4):
                rsz = cp.deepcopy(rsz[boundRect])
                rois.append(rsz)
    '''
    surnameX1 = 0
    surnameY1 = 0
    dniX1 = 0
    dniY1 = 0
    nameX2 = 0
    nameY2 = 0
    surnameX2 = 0
    surnameY2 = 0
    dniX2 = 0
    dniY2 = 0
    surnameX3 = 0
    surnameY3 = 0
    dniX3 = 0
    dniY3 = 0
    nameX3 = 0
    nameY3 = 0
    surnameX4 = 0
    surnameY4 = 0
    dniX4 = 0
    dniY4 = 0
    nameX4 = 0
    nameY4 = 0
    trobat = 0
    rectYo = 0
    rectYf = 0
    rectXo = 0
    rectXf = 0
    i = 0
    while (i < rowsV and (not trobat)):
        j = 0
        while (j < colsV and (not trobat)):
            if (vertexs[i][j] >= 5):
                #surnameX1 = j
                #surnameY1 = i
                #surnameY2 = i
                #surnameX3 = j
                #nameX1 = j
                #nameX3 = j
                rectYo = i
                trobat = 1
            j += 1
        i += 1
    if (trobat == 0): 
        print("Problema a l'hora de trobar les coordenades de les taules al primer bucle")
    trobat = 0
    i = rowsV - 1
    while (i >= 0 and (not trobat)):
        j = 0
        while (j < colsV and (not trobat)):
            if (vertexs[i][j] >= 5):
                #surnameX2 = j
                #surnameX4 = j
                #dniX2 = j
                #dniX4 = j
                rectYf = i
                trobat = 1
            j += 1
        i -= 1
    if (trobat == 0): 
        print("Problema a l'hora de trobar les coordenades de les taules al segon bucle")
    trobat = 0
    i = 0
    while (i < colsV and (not trobat)):
        j = 0
        while (j < rowsV and (not trobat)):
            #print("point ", j, " ",i, " :", vertexs[j][i])
            if (vertexs[j][i] >= 5):
                #surnameY3 = i
                #surnameY4 = i
                rectXo = i
                trobat = 1
            j += 1
        i += 1
    if (trobat == 0): 
        print("Problema a l'hora de trobar les coordenades de les taules al tercer bucle")
    trobat = 0
    i = colsV - 1
    while (i >=0 and (not trobat)):
        j = 0
        while (j < rowsV and (not trobat)):
            if (vertexs[j][i] >= 5):
                #surnameY3 = i
                #surnameY4 = i
                rectXf = i
                trobat = 1
            j += 1
        i -= 1
    if (trobat == 0): 
        print("Problema a l'hora de trobar les coordenades de les taules al quart bucle")
    print "First point surname: ",rectXo
    print "Second point surname: ",rectXf
    print "Third point surname: ",rectYo
    print "Fourth point surname: ",rectYf
    surname = cutImage(imgBitwise, rectYo,rectYf,rectXo,rectXf)
    cv2.imshow("Surname Table", surname)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    '''

    edges = cv2.Canny(imgBitwise,50,120)
    minLineLength = 200
    maxLineGap = 2
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("edges", edges)
    cv2.imshow("lines", imgBitwise)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    #th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    #cv2.imshow("img normal suavitazada invertidatest", img)
    #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #th4 = cv2.threshold(th3, 127, 255, cv2.THRESH_BINARY)
    #Suavitzen contorns
    #binarized = binarize(img, 248,1)
    
    #cv2.imshow("img normal suavitazada", binarized)

    #blur2 = cv2.GaussianBlur(binarized,(5,5),0)
    #cv2.imshow("img to find GAUS Blur", blur2)
    '''
    #Inverteix els valors -> Pasa la foto amb fons negre  i lletra blanca
    img = cv2.bitwise_not(binarized)
    #cv2.imshow("img normal suavitazada invertida", img)
    #cv2.imshow("img to find shapes gray", gray)
    #cv2.imshow("img to find shapes gray", img)
    #v2.imshow("img to find shapes binary", binarized)
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #cv2.imshow("img to find GAUS Blur Invertida", blur)
    #cv2.imshow("img to find GAUSS+++", th4)
    
    #Find Contours and mask everything that is not in the expected contour
    

    textFieldShapes, imageAuxiliar = findShapes(img, 4000, 60, 0.1)
    textFieldShapes.sort(key=sortContours)
    #textFliedshapes conte cada cuadricula
    cutImage(img, 0.06)
    print(imageAuxiliar)
    cv2.imshow("Imatge Retallada", imageAuxiliar)
    time.sleep(5)
    totalTextPrediction = ""
    for idx, s in enumerate(textFieldShapes[-3:]):
        boundingBoxes = lookForCharactersInImage(binarized, s)
        print "Found", len(boundingBoxes), "character shapes"
        
        model = models[1] if idx == 0 else models[0]
        img2 = cp.deepcopy(img)
        cv2.imshow("Imatge abans de prediccio", img2)
        img3 = cp.deepcopy(gray)
        cv2.imshow("Imatge grisos abans de prediccio", img3)
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
    '''



if __name__ == "__main__":
    main()
