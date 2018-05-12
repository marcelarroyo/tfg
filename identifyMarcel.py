#!/usr/bin/env python

# EXTERNAL DEPENDENCIES
import cv2
import numpy as np
from keras.models import load_model
from keras.datasets import mnist
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


class Exam:
    def __init__(self,numberDNI,predictionNumberDNI,letterDNI, predictionLetterDNI, name, predictionName, surname, predictionSurname, realFileName):
        self.numberDNI = numberDNI
        self.predictionNumberDNI = predictionNumberDNI
        self.letterDNI = letterDNI
        self.predictionLetterDNI = predictionLetterDNI
        self.name = name
        self.predictionName = predictionName
        self.surname = surname
        self.predictionSurname = predictionSurname
        self.realDNI = ""
        self.realName = ""
        self.realSurname = ""
        self.realFileName = realFileName
        self.realCsvIndex = -1

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


    fileName = []
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

            fullDNI = fields[3].upper()

            realName = fields[4].upper()

            realSurname = fields[5].upper()

            realDNI = fields[6].upper()

            realFullDNI = fields[7].upper()
            exam = []
            exam.append(dni)            
            exam.append(name)
            exam.append(surname)
            exam.append(ascii_name)
            exam.append(ascii_surname)
            exam.append(idx)
            exam.append(realName)
            exam.append(realSurname)
            exam.append(realDNI)
            exam.append(realFullDNI)
            exam.append(0)
            exam.append(fullDNI)
            fileName.append(ascii_name.upper()+ascii_surname.upper())
            data.append(exam)

    print data
    print "........................................."
    print fileName
    return data, fileName
    



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

def showExam(exam):
    print exam.numberDNI
    #self.predictionNumberDNI = predictionNumberDNI
    print exam.letterDNI
    #self.predictionLetterDNI = predictionLetterDNI
    print exam.name 
    #self.predictionName = predictionName
    print exam.surname
    #self.predictionSurname = predictionSurname
    print exam.realDNI
    print exam.realName
    print exam.realSurname
    print exam.realFileName
    print exam.realCsvIndex

def basicAssignment(exams,csvData):
    contador = 0
    elements = 0
    for exam in exams:
        numDNIString = ''.join(str(e) for e in exam.numberDNI)
        resultLetterDNI = DNILetter(int(numDNIString))
        print "Lletra calculada amb el numeros predits del DNI: ", resultLetterDNI
        print "Lletra predida del DNI: ", exam.letterDNI
        if (resultLetterDNI == exam.letterDNI):
            print "MATCH!!!!!!!!"
            contador += 1
            fullDNI = numDNIString + resultLetterDNI
            print "Full DNI: ", fullDNI
            trobat = 0
            i = 0
            print "Tamany de csvData",len(csvData)
            while (i < len(csvData) and not trobat):
                print csvData[i][0]
                print csvData[i][10]
                if (csvData[i][10] == 0 and csvData[i][11] == fullDNI):
                    csvData[i][10] = 1
                    exam.realDNI = csvData[i][11]
                    exam.realName = csvData[i][1]
                    exam.realSurname = csvData[i][2]
                    exam.realCsvIndex = csvData[i][5]
                    trobat = 1
                i += 1
        elements += 1
    print "Nombre total de assinacions directes al calcular la lletra: ", contador, " de ", elements
    return exams, csvData

def oneErrorAssignment (exams, csvData):
    for exam in exams:
        if (exam.realCsvIndex == -1):
            i = 0
            trobat = 0
            while (i < len(csvData) and not trobat):
                if (csvData[i][10] == 0):
                    j = 0
                    numDNIString = ''.join(str(e) for e in exam.numberDNI)
                    fullDNI = numDNIString + exam.letterDNI
                    onlyOneError = 0
                    errors = 0
                    while (j < len(fullDNI) and errors < 2):
                        if (fullDNI[j] != csvData[i][11][j]):
                            errors += 1
                        j += 1
                    if (errors < 2):
                        csvData[i][10] = 1
                        exam.realDNI = csvData[i][11]
                        exam.realName = csvData[i][1]
                        exam.realSurname = csvData[i][2]
                        exam.realCsvIndex = csvData[i][5]
                        trobat = 1
                i += 1
    return exams, csvData

def main():

    totalChecks=0
    totalHits=0
    checks=[0,0,0,0,0,0,0,0,0,0]
    hits=[0,0,0,0,0,0,0,0,0,0]
    fullHit = 0
    totalChecksDNILetter=0
    totalHitsDNILetter=0
    checksDNILetter=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    hitsDNILetter=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    totalChecksName=0
    totalHitsName=0
    checksName=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    hitsName=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    fullHitName = 0
    totalChecksSurname=0
    totalHitsSurname=0
    checksSurname=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    hitsSurname=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    fullHitSurname = 0

    modelMNIST = load_model("./modelAgusti28x28.h5")
    modelEMNIST = load_model("./model_emnist_28x28.h5")
    models = (modelMNIST, modelEMNIST)
    modelMNIST.summary()
    modelEMNIST.summary()

    images, nameList = parseArgs(sys.argv, "")
    names = parseNameList(nameList)
    exams = []
    results = []
    errors = []
    #print "ENTRO AL BUCLE"
    #print images
    for image in images:
	#print "HE ENTRAT AL BUCLE"
        #print (names[1])
        #fileResult, errorRates = processFile(image, names[1], models)
        totalChecksAux, totalHitsAux,checksAux, hitsAux, fullHitAux, totalChecksDNILetterAux, totalHitsDNILetterAux, checksDNILetterAux, hitsDNILetterAux, totalChecksNameAux, totalHitsNameAux, checksNameAux, hitsNameAux, fullHitNameAux, totalChecksSurnameAux, totalHitsSurnameAux, checksSurnameAux, hitsSurnameAux, fullHitSurnameAux, predictionExam = processFile(image, names[0], names[1], models)
        totalChecks += totalChecksAux
        totalHits += totalHitsAux
        checks=np.add(checksAux,checks)
        hits=np.add(hitsAux,hits)
        fullHit += fullHitAux
        totalChecksDNILetter += totalChecksDNILetterAux
        totalHitsDNILetter += totalHitsDNILetterAux
        checksDNILetter=np.add(checksDNILetter,checksDNILetterAux)
        hitsDNILetter=np.add(hitsDNILetter,hitsDNILetterAux)
        totalChecksName += totalChecksNameAux
        totalHitsName += totalHitsNameAux
        checksName=np.add(checksNameAux,checksName)
        hitsName=np.add(hitsName,hitsNameAux)
        fullHitName+=fullHitNameAux
        totalChecksSurname += totalChecksSurnameAux
        totalHitsSurname += totalHitsSurnameAux
        checksSurname=np.add(checksSurnameAux,checksSurname)
        hitsSurname=np.add(hitsSurname,hitsSurnameAux)
        fullHitSurname+=fullHitSurnameAux
        exams.append(predictionExam)
        print "Total hits DNI(NUM)fins al moment: ",totalHits
        print "Total checks DNI(NUM)fins al moment: ",totalChecks
        print "Total hits NOM fins al moment: ",totalHitsName
        print "Total checks NOM fins al moment: ",totalChecksName
        print "Total hits COGNOM fins al moment: ",totalHitsSurname
        print "Total checks COGNOM fins al moment: ",totalChecksSurname
        #results.append(fileResult)
        #errors.append(errorRates)
    #Mostra el % d'encerts total
    print "-------------------DNI--------------------"
    print "HITS DNI (NUM): ",totalHits
    print "CHECKS DNI (NUM): ",totalChecks
    totalPercentatge = totalHits/float(totalChecks)
    print "PERCENTATGE ENCERTS DNI (NUM): ", totalPercentatge
    print "ENCERTS DNI COMPLERTS (NUM): ", fullHit
    #Mostra el percentatge d'encerts per a cada numero
    i = 0
    while (i < 10):
        numberPercentatge = hits[i]/float(checks[i])
        print "Percentat d'encert sobre el nombre ",i," : ", numberPercentatge
        i += 1
    #Mostra el % d'encerts total lletra dni
    print "HITS DNI (LLETRA): ",totalHitsDNILetter
    print "CHECKS DNI (LLETRA): ",totalChecksDNILetter
    totalPercentatgeDNILetter = totalHitsDNILetter/float(totalChecksDNILetter)
    print "PERCENTATGE ENCERTS DNI (LLETRA): ", totalPercentatgeDNILetter
    #Mostra el percentatge d'encerts per a cada lletra
    i = 0
    while (i < 26):
        numberPercentatge = hitsDNILetter[i]/float(checksDNILetter[i])
        print "Percentat d'encert sobre la lletra ",str(unichr(i + 65))," : ", numberPercentatge
        i += 1
    print "-------------------NOM--------------------"
    print "HITS NOM : ",totalHitsName
    print "CHECKS NOM: ",totalChecksName
    totalPercentatgeName = totalHitsName/float(totalChecksName)
    print "PERCENTATGE ENCERTS NOM: ", totalPercentatgeName
    print "ENCERTS NOM COMPLERTS: ", fullHitName
    #Mostra el percentatge d'encerts per a cada lletra
    i = 0
    while (i < 26):
        numberPercentatge = hitsName[i]/float(checksName[i])
        print "Percentat d'encert en el NOM sobre la lletra ",str(unichr(i + 65))," : ", hitsName[i]," / ",checksName[i],": ", numberPercentatge
        i += 1

    print "-------------------COGNOM--------------------"
    print "HITS COGNOM : ",totalHitsSurname
    print "CHECKS COGNOM: ",totalChecksSurname
    totalPercentatgeSurname = totalHitsSurname/float(totalChecksSurname)
    print "PERCENTATGE ENCERTS COGNOM: ", totalPercentatgeSurname
    print "ENCERTS COGNOM COMPLERTS: ", fullHitSurname
    #Mostra el percentatge d'encerts per a cada lletra
    i = 0
    while (i < 26):
        numberPercentatge = hitsName[i]/float(checksName[i])
        print "Percentat d'encert en el NOM sobre la lletra ",str(unichr(i + 65))," : ", hitsName[i]," / ",checksName[i],": ", numberPercentatge
        i += 1

    #Inici de assignacions
    examsBasic, namesBasic = basicAssignment(exams,names[0])
    examsOneError, namesOneError = oneErrorAssignment(examsBasic,namesBasic)
    identifiedExams = 0
    for exam in examsOneError:
        showExam(exam)
        if (exam.realCsvIndex != -1):
            identifiedExams += 1
    print "EXAMENS IDENTIFICATS: ", identifiedExams
    '''
    result,unused, unrecognizable = mapResults(results, names[0], images, errors)
    fin = {
        "identified" : result, 
        "unused" : unused, 
        "unrecognizable" : unrecognizable
    }

    json.dump(fin, sys.stdout)
'''
def isEmpty(image):
    ret = 1
    cols = image.shape[1]
    rows = image.shape[0]
    cont = 0
    i = 0

    while (i < rows):
        j = 0
        while (j < cols):
            if (image[i][j] == 255):
                cont += 1
            j += 1
        i += 1
    if (cont > 20):
        ret = 0
    else:
        ret = 1
        #cv2.imshow("isEmpty", image)
        #cv2.waitKey(1000)
        #cv2.destroyAllWindows()
    return ret

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


def DNILetter(numeros):
    return "TRWAGMYFPDXBNJZSQVHLCKE"[numeros%23]

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

def processFile(filename, data ,possibleWords, models):
    totalChecks=0
    totalHits=0
    checks=[0,0,0,0,0,0,0,0,0,0]
    hits=[0,0,0,0,0,0,0,0,0,0]
    fullHit = 0
    totalChecksDNILetter=0
    totalHitsDNILetter=0
    checksDNILetter=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    hitsDNILetter=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    totalChecksName = 0
    totalHitsName = 0
    checksName = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    hitsName = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    fullHitName = 0
    totalChecksSurname = 0
    totalHitsSurname = 0
    checksSurname = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    hitsSurname = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    fullHitSurname = 0
    prediction_dni = []
    prediction_letter = []
    prediction_name = []
    prediction_surname = []
    predicted_dni = []
    predicted_letter = ""
    predicted_name = []
    predicted_surname = []
    print filename
    nameNoPath = os.path.basename(filename)
    print nameNoPath
    (name,ext) = os.path.splitext(nameNoPath)
    print name 
    #Extreure index de la imatge que s'esta tractant
    if name in possibleWords:
        indexImage = possibleWords.index(name)
        #mostrar les dades de la imatge que s'esta mostrant
        print data[indexImage]
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
        #cv2.imwrite('marcelo.png',source_dilated)
        #cv2.destroyAllWindows()
        #Busca rectes verticals amb una distancia major a 50 pixels
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize));
        source_eroded2 = cv2.erode(vertical, verticalStructure, iterations=1);
        source_dilated2 = cv2.dilate(source_eroded2,verticalStructure, iterations=1);
        #cv2.imshow("vertical", source_dilated2)
        #cv2.waitKey(10000)
        #cv2.imwrite('marcelo1.png',source_dilated2)
        #cv2.destroyAllWindows()
        #Imprimim les rectes horitzontals i verticals
        table = source_dilated2 + source_dilated
        #cv2.imshow("Table", table)
        #cv2.waitKey(10000)
        #cv2.destroyAllWindows()
        #Extreiem els vertexs de la taula
        vertexs = cv2.bitwise_and(source_dilated, source_dilated2)
        #cv2.imwrite('marcelo2.png',vertexs)
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
        #cv2.imwrite('marcelo3.png',cleanTable)
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
        trobat = 0
        i = 0
        while (i < rowsV and (not trobat)):
            j = 0
            while (j < colsV and (not trobat)):
                if (vertexs[i][j] >= 5):
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
        contourTable = cutImage(imgBitwise, rectYo,rectYf,rectXo,rectXf)
        #cv2.imshow("Contour Table", contourTable)
        #cv2.waitKey(10000)
        #cv2.destroyAllWindows()
        #Separar les els diferents camps de la taula
        rows = contourTable.shape[0]
        cols = contourTable.shape[1]
        surname = cp.deepcopy(cutImage(contourTable, 0, 60, 0, cols - 1))
        #cv2.imshow("Surname Table", surname)
        #cv2.waitKey(10000)
        #cv2.destroyAllWindows()
        name = cp.deepcopy(cutImage(contourTable, rows-61, rows-1, 0, 666))
        #cv2.imshow("Name Table", name)
        #cv2.waitKey(10000)
        #cv2.destroyAllWindows()
        dni = cp.deepcopy(cutImage(contourTable, rows-61, rows-1, 752, cols - 1))
        #cv2.imshow("DNI Table", dni)
        #cv2.waitKey(10000)
        #cv2.destroyAllWindows()
        #Separar cada caracter en una imatge
        dni_images = []
        name_images = []
        surname_images = []
        dni_image_size = 44
        name_image_size = 44
        surname_image_size = 44
        #print("DNI size: ", dni_image_size)
        #print("name size: ", name_image_size)
        #print("surname size: ", surname_image_size)
        i = 0
        while (i < 26):
            if (i < 9):
                dni_images.append(cp.deepcopy(cutImage(dni,0,dni.shape[0]-1,i*44, i*44 + 45)))
                #cv2.imshow("DNI Images", dni_images[i])
                #cv2.waitKey(10000)
                #cv2.destroyAllWindows()
            if (i < 15):
                name_images.append(cp.deepcopy(cutImage(name,0,name.shape[0]-1,i*44, i*44 + 45)))
                #cv2.imshow("Name Images", name_images[i])
                #cv2.waitKey(10000)
                #cv2.destroyAllWindows()
            if (i < 26):
                surname_images.append(cp.deepcopy(cutImage(surname,0,surname.shape[0]-1,i*44, i*44 + 45)))
                #cv2.imshow("Surname Images", surname_images[i])
                #cv2.waitKey(10000)
                #cv2.destroyAllWindows()
            i += 1
        i = 0
        dni_images_28_28 = []
        while(i < 9):
            #Modifiquem la imatge a 28x28
            #cv2.imshow("Number DNI Abans de retallar", dni_images[i])
            #cv2.waitKey(10000)
            #cv2.destroyAllWindows()
            colsAux = dni_images[i].shape[1]
            rowsAux = dni_images[i].shape[0]
            #Selecionem els marges a retallar
            top = int(round(0.15*rowsAux))
            bottom = int(round(0.10*rowsAux))
            left = int(round(0.10*colsAux))
            right = int(round(0.10*colsAux))

            #Pintar de negre la part amb ratlles superior
            j = 0
            while (j < top):
                k = 0
                while (k < colsAux):
                    dni_images[i][j][k] = 0
                    k += 1
                j += 1
            #Pintar de negre la part amb ratlles esquerre
            j = 0
            while (j < rowsAux):
                k = 0
                while (k < left):
                    dni_images[i][j][k] = 0
                    k += 1
                j += 1
            #Pintar de negre la part amb ratlles inferior
            j = rowsAux - bottom
            while (j < rowsAux):
                k = 0
                while (k < colsAux):
                    dni_images[i][j][k] = 0
                    k += 1
                j += 1
            #Pintar de negre la part amb ratlles dreta
            j = 0
            while (j < rowsAux):
                k = colsAux - (right + 1)
                while(k < colsAux):
                    dni_images[i][j][k] = 0
                    k += 1
                j += 1



            #aux1 = dni_images[i][0+top:rowsAux-bottom, 0+left:colsAux-right]

            #cv2.imshow("Number DNI Cropped", dni_images[i])
            #cv2.waitKey(10000)
            #cv2.destroyAllWindows()
            '''
            #Normalitzar la imatge
            x = 0
            while (x < aux1.shape[0]):
                y = 0
                while(y < aux1.shape[1]):
                    if (aux1[x][y] > 30):
                        aux1[x][y] = 0
                    else: 
                        aux1[x][y] =  255
                    y += 1
                x += 1
            cv2.imshow("Number DNI  Maximize", aux1)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()

            '''

            #Resize equitatiu sense deformacions (afegir pixels negres per fer la imatge 59*59)
            diferenceDimensions = rowsAux - colsAux
            if (diferenceDimensions%2 == 1):
                pixelsLeft = int(round(diferenceDimensions/2)) + 1
                pixelsRight = int(round(diferenceDimensions/2))
            else:
                pixelsLeft = diferenceDimensions/2
                pixelsRight = diferenceDimensions/2
            print ("columnes sumades: ", pixelsLeft + colsAux + pixelsRight)
            print ("Files: ", rowsAux)
            #Crear una image de 59*59
            aux59x59 = cv2.resize(dni_images[i], (rowsAux, pixelsLeft + colsAux + pixelsRight), cv2.INTER_LINEAR)
            #Afegir el marges negres a la imatge original
            whiteColor = 0
            j = 0 
            while (j < rowsAux):
                k = 0
                while (k < pixelsLeft):
                    aux59x59[j][k] = 0
                    k += 1
                j += 1
            j = 0
            while (j < rowsAux):
                k = 0
                while (k < colsAux):
                    aux59x59[j][k + pixelsLeft] = dni_images[i][j][k]
                    if (dni_images[i][j][k] > whiteColor):
                        whiteColor = dni_images[i][j][k]
                    k += 1
                j += 1
            j = 0
            while (j < rowsAux):
                k = colsAux + pixelsLeft
                while (k < colsAux + pixelsRight):
                    aux59x59[j][k] = 0
                    k += 1
                j += 1

            #cv2.imshow("Redimensionament equitatiu", aux59x59)
            #cv2.waitKey(10000)
            #cv2.destroyAllWindows()

            #Correccio de color
            j = 0
            while (j < aux59x59.shape[0]):
                k = 0
                while (k < aux59x59.shape[1]):
                    aux59x59[j][k] = (aux59x59[j][k] * 255) / whiteColor
                    if (aux59x59[j][k] < 45):
                        aux59x59[j][k] = 0
                    else: 
                        aux59x59[j][k] = 255
                    k += 1
                j += 1

            #cv2.imshow("Number DNI Corecction Color", aux59x59)
            #cv2.waitKey(10000)
            #cv2.destroyAllWindows()

            #Resize image
            #aux2 = cv2.resize(aux59x59, (64, 64), cv2.INTER_LINEAR)
            aux2 = cv2.resize(aux59x59, (28, 28), cv2.INTER_LINEAR)
            #print aux.shape[0]
            print aux2.shape
            #aux = cv2.resize(aux,None,fx=2,fy=2, cv2.INTER_LINEAR)
            #colsAux = aux.shape[1]
            #rowsAux = aux.shape[0]
            #cv2.imshow("Number DNI Maximized & Cropped", aux2)
            #cv2.waitKey(1000)
            #cv2.destroyAllWindows()


            '''
            x = 0
            while (x < 28):
                y = 0
                while(y < 28):
                    if (aux[x][y] > 30):
                        aux[x][y] = 0
                    else:
                        aux[x][y] = 1
                    y += 1
                x += 1
            cv2.imshow("Number DNI Reescaled & Cropped & Normalize", aux)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()
            '''
            #Apliquem marges negres per eliminar les ralles de la taula
            '''
            horizontalStructureAux = cv2.getStructuringElement(cv2.MORPH_RECT, (28,1));
            source_eroded_aux = cv2.erode(aux, horizontalStructureAux, iterations=1);
            source_dilated_aux = cv2.dilate(source_eroded_aux,horizontalStructureAux, iterations=1);
            cv2.imshow("horizontal_aux", source_dilated_aux)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()

            '''


            '''
            top = int(round(0.05*rowsAux))
            bottom = int(round(0.05*rowsAux))
            left = int(round(0.10*colsAux))
            right = int(round(0.10*colsAux))
            borderImage = cv2.copyMakeBorder(aux, top, bottom, left, right,cv2.BORDER_CONSTANT ,0)
            cv2.imshow("Number DNI Reescaled & Border", borderImage)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()
            '''
            #Normalitzem la imatge
            '''
            z = 0
            while (z < 28):
                w = 0
                while (w < 28):
                    print aux[z][w],
                    w+=1
                z+=1
                print
    '''
            #Convertim l'estructura en la estructura per defecte d'entrada
           # arr = np.asarray(aux2)
            #arr = np.expand_dims(arr, axis=0)
            #arr = np.expand_dims(arr, axis=3)
            '''
            print res
            z = 0
            while (z < 28):
                w = 0
                while (w < 28):
                    print res[z][w][0],
                    w+=1
                z+=1
                print
            '''
            #res = res/255
            '''
            z = 0
            while (z < 28):
                w = 0
                while (w < 28):
                    print res[0][0][z][w],
                    w+=1
                z+=1
                print
            '''
            #Predim el caracter
            aux2= aux2/255
            res= aux2.reshape(1,28,28,1)
            if (i != 8):
                print models[0].predict(res)
                predictionNumber = np.argmax(models[0].predict(res))
                print "Resultat de la prediccio: ", predictionNumber
                print "Numero real mostrat: ", data[indexImage][8][i]
                #Comprovar si es el nombre correcte
                if (data[indexImage][8][i] != '*' and ord(data[indexImage][8][i]) >= 48 and ord(data[indexImage][8][i]) <= 57):
                    totalChecks +=1
                    checks[int(data[indexImage][8][i])] +=1
                    if (int(data[indexImage][8][i]) == predictionNumber):
                        hits[int(data[indexImage][8][i])] +=1
                        totalHits +=1

                print "prediccions realitxzades fins el momment: ", totalChecks
                print "Encerts fins el moment: ", totalHits
                #prediction_dni.append(models[1].predict(arr)[0])
                prediction_dni.append(models[0].predict(res))
                predicted_dni.append(predictionNumber)
            else:
                print models[1].predict(res)
                predictionLetter = np.argmax(models[1].predict(res))
                print "Resultat de la prediccio: ", str(unichr(predictionLetter+65))
                print "Numero real mostrat: ", data[indexImage][9][i]
                #Comprovar si es el nombre correcte
                if (data[indexImage][9][i] != '*' and ((ord(data[indexImage][9][i]) >= 65 and ord(data[indexImage][9][i]) <= 90) or (ord(data[indexImage][9][i]) >= 97 and ord(data[indexImage][9][i]) <= 122))):
                    totalChecksDNILetter +=1
                    checksDNILetter[ord(data[indexImage][9][i])-65] +=1
                    if ((ord(data[indexImage][9][i])-65) == predictionLetter):
                        hitsDNILetter[ord(data[indexImage][9][i])-65] +=1
                        totalHitsDNILetter +=1

                print "prediccions realitxzades fins el momment: ", totalChecksDNILetter
                print "Encerts fins el moment: ", totalHitsDNILetter
                #prediction_dni.append(models[1].predict(arr)[0])
                prediction_letter.append(models[1].predict(res))
                predicted_letter = str(unichr(predictionLetter+65))

            i += 1
        i = 0
        itCaracter = 0
        while(i < 15):
            #Modifiquem la imatge a 28x28
            
            colsAux = name_images[i].shape[1]
            rowsAux = name_images[i].shape[0]
            #Selecionem els marges a retallar
            top = int(round(0.15*rowsAux))
            bottom = int(round(0.10*rowsAux))
            left = int(round(0.15*colsAux))
            right = int(round(0.10*colsAux))

            #Pintar de negre la part amb ratlles superior
            j = 0
            while (j < top):
                k = 0
                while (k < colsAux):
                    name_images[i][j][k] = 0
                    k += 1
                j += 1
            #Pintar de negre la part amb ratlles esquerre
            j = 0
            while (j < rowsAux):
                k = 0
                while (k < left):
                    name_images[i][j][k] = 0
                    k += 1
                j += 1
            #Pintar de negre la part amb ratlles inferior
            j = rowsAux - bottom
            while (j < rowsAux):
                k = 0
                while (k < colsAux):
                    name_images[i][j][k] = 0
                    k += 1
                j += 1
            #Pintar de negre la part amb ratlles dreta
            j = 0
            while (j < rowsAux):
                k = colsAux - (right + 1)
                while(k < colsAux):
                    name_images[i][j][k] = 0
                    k += 1
                j += 1


            #Resize equitatiu sense deformacions (afegir pixels negres per fer la imatge 59*59)
            diferenceDimensions = rowsAux - colsAux
            if (diferenceDimensions%2 == 1):
                pixelsLeft = int(round(diferenceDimensions/2)) + 1
                pixelsRight = int(round(diferenceDimensions/2))
            else:
                pixelsLeft = diferenceDimensions/2
                pixelsRight = diferenceDimensions/2
            print ("columnes sumades: ", pixelsLeft + colsAux + pixelsRight)
            print ("Files: ", rowsAux)
            #Crear una image de 59*59
            aux59x59 = cv2.resize(name_images[i], (rowsAux, pixelsLeft + colsAux + pixelsRight), cv2.INTER_LINEAR)
            #Afegir el marges negres a la imatge original
            whiteColor = 0
            j = 0 
            while (j < rowsAux):
                k = 0
                while (k < pixelsLeft):
                    aux59x59[j][k] = 0
                    k += 1
                j += 1
            j = 0
            while (j < rowsAux):
                k = 0
                while (k < colsAux):
                    aux59x59[j][k + pixelsLeft] = name_images[i][j][k]
                    if (name_images[i][j][k] > whiteColor):
                        whiteColor = name_images[i][j][k]
                    k += 1
                j += 1
            j = 0
            while (j < rowsAux):
                k = colsAux + pixelsLeft
                while (k < colsAux + pixelsRight):
                    aux59x59[j][k] = 0
                    k += 1
                j += 1
            #Threshold per veure si hi ha alguna linea a limatge de colro blanc
            if (whiteColor > 50):
                #Correccio de color
                j = 0
                while (j < aux59x59.shape[0]):
                    k = 0
                    while (k < aux59x59.shape[1]):
                        aux59x59[j][k] = (aux59x59[j][k] * 255) / whiteColor
                        if (aux59x59[j][k] < 40):
                            aux59x59[j][k] = 0
                        else: 
                            aux59x59[j][k] = 255
                        k += 1
                    j += 1

            #cv2.imshow("Number DNI Corecction Color", aux59x59)
            #cv2.waitKey(10000)
            #cv2.destroyAllWindows()

            #Resize image
            #aux2 = cv2.resize(aux59x59, (64, 64), cv2.INTER_LINEAR)
            aux2 = cv2.resize(aux59x59, (28, 28), cv2.INTER_LINEAR)
            #cv2.imshow("Number DNI Corecction Color", aux2)
            #cv2.waitKey(1000)
            #cv2.destroyAllWindows()


            if (not isEmpty(aux2)):
                #print aux.shape[0]
                print aux2.shape

                #Predim el caracter
                aux2= aux2/255
                res= aux2.reshape(1,28,28,1)

                print models[1].predict(res)
                predictionLetter = np.argmax(models[1].predict(res))
                print "Resultat de la prediccio: ", str(unichr(predictionLetter+65))
                print "Numero real mostrat: ", data[indexImage][6][itCaracter]
                #Comprovar si es el nombre correcte
                if (data[indexImage][6][itCaracter] != '*' and ((ord(data[indexImage][6][itCaracter]) >= 65 and ord(data[indexImage][6][itCaracter]) <= 90) or (ord(data[indexImage][6][itCaracter]) >= 97 and ord(data[indexImage][6][itCaracter]) <= 122))):
                    totalChecksName +=1
                    checksName[ord(data[indexImage][6][itCaracter])-65] +=1
                    if ((ord(data[indexImage][6][itCaracter])-65) == predictionLetter):
                        hitsName[ord(data[indexImage][6][itCaracter])-65] +=1
                        totalHitsName +=1

                print "prediccions realitxzades fins el momment NOM: ", totalChecksName
                print "Encerts fins el moment NOM: ", totalHitsName
                #prediction_dni.append(models[1].predict(arr)[0])
                prediction_name.append(models[1].predict(res))
                predicted_name.append(str(unichr(predictionLetter+65)))
                itCaracter += 1
            #else: No incrementar literador del del comprovant
            i += 1
        i = 0
        itCaracterSurname = 0
        while(i < 26):
            #Modifiquem la imatge a 28x28
            
            colsAux = surname_images[i].shape[1]
            rowsAux = surname_images[i].shape[0]
            #Selecionem els marges a retallar
            top = int(round(0.15*rowsAux))
            bottom = int(round(0.10*rowsAux))
            left = int(round(0.20*colsAux))
            right = int(round(0.10*colsAux))

            #Pintar de negre la part amb ratlles superior
            j = 0
            while (j < top):
                k = 0
                while (k < colsAux):
                    surname_images[i][j][k] = 0
                    k += 1
                j += 1
            #Pintar de negre la part amb ratlles esquerre
            j = 0
            while (j < rowsAux):
                k = 0
                while (k < left):
                    surname_images[i][j][k] = 0
                    k += 1
                j += 1
            #Pintar de negre la part amb ratlles inferior
            j = rowsAux - bottom
            while (j < rowsAux):
                k = 0
                while (k < colsAux):
                    surname_images[i][j][k] = 0
                    k += 1
                j += 1
            #Pintar de negre la part amb ratlles dreta
            j = 0
            while (j < rowsAux):
                k = colsAux - (right + 1)
                while(k < colsAux):
                    surname_images[i][j][k] = 0
                    k += 1
                j += 1


            #Resize equitatiu sense deformacions (afegir pixels negres per fer la imatge 59*59)
            diferenceDimensions = rowsAux - colsAux
            if (diferenceDimensions%2 == 1):
                pixelsLeft = int(round(diferenceDimensions/2)) + 1
                pixelsRight = int(round(diferenceDimensions/2))
            else:
                pixelsLeft = diferenceDimensions/2
                pixelsRight = diferenceDimensions/2
            print ("columnes sumades: ", pixelsLeft + colsAux + pixelsRight)
            print ("Files: ", rowsAux)
            #Crear una image de 59*59
            aux59x59 = cv2.resize(surname_images[i], (rowsAux, pixelsLeft + colsAux + pixelsRight), cv2.INTER_LINEAR)
            #Afegir el marges negres a la imatge original
            whiteColor = 0
            j = 0 
            while (j < rowsAux):
                k = 0
                while (k < pixelsLeft):
                    aux59x59[j][k] = 0
                    k += 1
                j += 1
            j = 0
            while (j < rowsAux):
                k = 0
                while (k < colsAux):
                    aux59x59[j][k + pixelsLeft] = surname_images[i][j][k]
                    if (surname_images[i][j][k] > whiteColor):
                        whiteColor = surname_images[i][j][k]
                    k += 1
                j += 1
            j = 0
            while (j < rowsAux):
                k = colsAux + pixelsLeft
                while (k < colsAux + pixelsRight):
                    aux59x59[j][k] = 0
                    k += 1
                j += 1
            #Threshold per veure si hi ha alguna linea a limatge de colro blanc
            if (whiteColor > 50):
                #Correccio de color
                j = 0
                while (j < aux59x59.shape[0]):
                    k = 0
                    while (k < aux59x59.shape[1]):
                        aux59x59[j][k] = (aux59x59[j][k] * 255) / whiteColor
                        if (aux59x59[j][k] < 40):
                            aux59x59[j][k] = 0
                        else: 
                            aux59x59[j][k] = 255
                        k += 1
                    j += 1

            #cv2.imshow("Number DNI Corecction Color", aux59x59)
            #cv2.waitKey(10000)
            #cv2.destroyAllWindows()

            #Resize image
            #aux2 = cv2.resize(aux59x59, (64, 64), cv2.INTER_LINEAR)
            aux2 = cv2.resize(aux59x59, (28, 28), cv2.INTER_LINEAR)

            if (not isEmpty(aux2)):
                #print aux.shape[0]
                print aux2.shape

                #Predim el caracter
                aux2= aux2/255
                res= aux2.reshape(1,28,28,1)

                print models[1].predict(res)
                predictionLetter = np.argmax(models[1].predict(res))
                print "Resultat de la prediccio: ", str(unichr(predictionLetter+65))
                print "Numero real mostrat: ", data[indexImage][7][itCaracterSurname]
                #Comprovar si es el nombre correcte
                if (data[indexImage][7][itCaracterSurname] != '*' and ((ord(data[indexImage][7][itCaracterSurname]) >= 65 and ord(data[indexImage][7][itCaracterSurname]) <= 90) or (ord(data[indexImage][7][itCaracterSurname]) >= 97 and ord(data[indexImage][7][itCaracterSurname]) <= 122))):
                    totalChecksSurname +=1
                    checksSurname[ord(data[indexImage][7][itCaracterSurname])-65] +=1
                    if ((ord(data[indexImage][7][itCaracterSurname])-65) == predictionLetter):
                        hitsSurname[ord(data[indexImage][7][itCaracterSurname])-65] +=1
                        totalHitsSurname +=1

                print "prediccions realitxzades fins el momment COGNOM: ", totalChecksSurname
                print "Encerts fins el moment COGNOM: ", totalHitsSurname
                #prediction_dni.append(models[1].predict(arr)[0])
                prediction_surname.append(models[1].predict(res))
                predicted_surname.append(str(unichr(predictionLetter+65)))
                itCaracterSurname += 1
            #else: No incrementar literador del del comprovant
            i += 1
        '''
        #Assignacio directa en cas de que la lletra calculada sigui igual que la lletra predida
        numDNIString = ''.join(str(e) for e in predicted_dni)
        resultLetterDNI = letterDNI(int(numDNIString))
        print "Lletra calculada amb el numeros predits del DNI: ", resultLetterDNI
        print "Lletra predida del DNI: ", predicted_letter
        '''
        resultPrediction = Exam(predicted_dni, prediction_dni, predicted_letter, prediction_letter, predicted_name, prediction_name, predicted_surname, prediction_surname, nameNoPath)
        #resultLetterDNI = letterDNI(''.join(str(e) for e in prediction_dni))
        #print resultLetterDNI
        if totalHits == 8:
            fullHit = 1
        else:
            fullHit = 0
        if totalHitsName == totalChecksName:
            fullHitName = 1
        else:
            fullHitName = 0
        if totalHitsSurname == totalChecksSurname:
            fullHitSurname = 1
        else:
            fullHitSurname = 0
    return totalChecks, totalHits, checks, hits, fullHit, totalChecksDNILetter, totalHitsDNILetter, checksDNILetter, hitsDNILetter, totalChecksName, totalHitsName, checksName, hitsName, fullHitName, totalChecksSurname, totalHitsSurname, checksSurname, hitsSurname, fullHitSurname, resultPrediction

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
