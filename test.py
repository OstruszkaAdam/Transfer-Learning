# test.py

import time
import shutil
import os
import tensorflow as tf
import numpy as np
import cv2

from utils.zapsatJakoExcel import zapsatPolePoliJakoExcel

# module-level variables ##############################################################################################
TRAINING_OUTPUT_DIR = os.getcwd() + "/3_training_output"

RETRAINED_LABELS_TXT_FILE_LOC = TRAINING_OUTPUT_DIR + "/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = TRAINING_OUTPUT_DIR + "/" + "retrained_graph.pb"

# where to save summary logs for TensorBoard
TENSORBOARD_TEST_LOGS_DIR = os.getcwd() + '/' + '5_test_chache/tensorboard_logs'

TEST_INPUT_IMAGES_DIR = os.getcwd() + "/4_test_input_images"
TEST_OUTPUT_DIR = os.getcwd() + "/6_test_output/"

# definovani, jakou barvu maji mit popisky fotek
# POZOR: v OpenCV nejsou barvy v pořadí RGB, ale BGR
LABEL_FONT_COLOR = (255.0, 255.0, 255.0)   # bila
# LABEL_FONT_COLOR = (0.0, 165.0, 255.0)   # oranzova
# LABEL_FONT_COLOR = (0.0, 0.0, 255.0)     # cervena
# LABEL_FONT_COLOR = (255.0, 0.0, 0.0)     # modra

# pokud je nastaveno na False, otestuje program vsechny snimky najednou
# pokud je nastaveno na True, program pred prechodem na dalsi snimek ceka na stisknuti libovolne klavesy
PROCHAZET_SNIMKY_JEDNOTLIVE = False

#######################################################################################################################

def main():
    # aktualni cas a datum pro pouziti v nazvech vystupnich souboru
    timeStamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # vytvoreni slozky na ukladani vystupu z konzole
    outputSubirectoryName = TEST_OUTPUT_DIR + "/" + timeStamp + "_test_output_results"
    tf.gfile.MakeDirs(outputSubirectoryName)

    # vytvoreni nazvu souboru s vystupy testu
    nazevExcelovskehoSouboru = outputSubirectoryName + "/" + timeStamp + "_test_output_results"

    print("probiha spousteni programu . . .")

    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

    # get a list of classifications from the labels file
    classifications = []
    # for each line in the label file . . .
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        # remove the carriage return
        classification = currentLine.rstrip()
        # and append to the list
        classifications.append(classification)
    # end for

    # show the classifications to prove out that we were able to read the label file successfully
    print("kategorie, mezi kterymi program muze vybirat = " + str(classifications))

    # load the graph from file
    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        # instantiate a GraphDef object
        graphDef = tf.GraphDef()
        # read in retrained graph into the GraphDef object
        graphDef.ParseFromString(retrainedGraphFile.read())
        # import the graph into the current default Graph, note that we don't need to be concerned with the return value
        _ = tf.import_graph_def(graphDef, name='')
    # end with

    # if the test image directory listed above is not valid, show an error message and bail
    if not os.path.isdir(TEST_INPUT_IMAGES_DIR):
        print("the test image directory does not seem to be a valid directory, check file / directory paths")
        return
    # end if

    # inicializace pole pro ukladani vysledku testu
    vsechnyRadkyKzapisuDoExcelu = []
    zahlaviKzapisuDoExcelu = ["nazev souboru", "zarazeno do kategorie", "se spolehlivosti [%]", "kat1", "kat2", "kat3", "kat4", "kat5", "kat6"]

    with tf.Session() as sess:

        # for each file in the test images directory . . .
        for fileName in os.listdir(TEST_INPUT_IMAGES_DIR):
            # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
            if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                continue
            # end if

            print("-----------------------------------------------------------------")
            jedenRadekKzapisuDoExcelu = []
            # show the file name on std out
            print("zpracovava se soubor " + fileName)
            jedenRadekKzapisuDoExcelu.append(fileName)

            # get the file name and full path of the current image file
            imageFileWithPath = os.path.join(TEST_INPUT_IMAGES_DIR, fileName)
            # attempt to open the image with OpenCV
            openCVImage = cv2.imread(imageFileWithPath)

            # if we were not able to successfully open the image, continue with the next iteration of the for loop
            if openCVImage is None:
                print("unable to open " + fileName + " as an OpenCV image")
                jedenRadekKzapisuDoExcelu.append("unable to open")
                continue
            # end if

            # get the final tensor from the graph
            finalTensor = sess.graph.get_tensor_by_name('final_result:0')

            # convert the OpenCV image (numpy array) to a TensorFlow image
            tfImage = np.array(openCVImage)[:, :, 0:3]

            # run the network to get the predictions
            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

            # sort predictions from most confidence to least confidence
            sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

            # keep track of if we're going through the next for loop for the first time so we can show more info about
            # the first prediction, which is the most likely prediction (they were sorted descending above)
            onMostLikelyPrediction = True
            # for each prediction . . .
            for prediction in sortedPredictions:
                strClassification = classifications[prediction]

                # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
                if strClassification.endswith("s"):
                    strClassification = strClassification[:-1]
                # end if

                # get confidence, then get confidence rounded to 2 places after the decimal
                confidence = predictions[0][prediction]

                # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
                if onMostLikelyPrediction:
                    # get the score as a %
                    scoreAsAPercent = confidence * 100.0
                    # show the result to std out
                    print("program zaradil snimek do kategorie " + strClassification + " (" + "{0:.2f}".format(scoreAsAPercent) + "% confidence)")
                    jedenRadekKzapisuDoExcelu.append(strClassification)
                    jedenRadekKzapisuDoExcelu.append("{0:.2f}".format(scoreAsAPercent))
                    # write the result on the image
                    writeResultOnImage(outputSubirectoryName,openCVImage, timeStamp, fileName, strClassification + " (" + "{0:.2f}".format(scoreAsAPercent) + "% confidence)")
                    # finally we can show the OpenCV image
                    cv2.imshow(fileName, openCVImage)
                    # mark that we've show the most likely prediction at this point so the additional information in
                    # this if statement does not show again for this image
                    onMostLikelyPrediction = False
                # end if

                # for any prediction, show the confidence as a ratio to five decimal places
                print(strClassification + " (" +  "{0:.5f}".format(confidence) + ")")
                jedenRadekKzapisuDoExcelu.append(strClassification + " (" + "{0:.5f}".format(confidence) + ")")

            # end for
            print("\n")

            vsechnyRadkyKzapisuDoExcelu.append(jedenRadekKzapisuDoExcelu)

            # pause until a key is pressed so the user can see the current image (shown above) and the prediction info
            if PROCHAZET_SNIMKY_JEDNOTLIVE:
                cv2.waitKey()
            # after a key is pressed, close the current window to prep for the next time around
            cv2.destroyAllWindows()
        # end for
    # end with

    # prepare necessary directories that can be used during training
    prepare_file_system()

    # write the graph to file so we can view with TensorBoard
    tfFileWriter = tf.summary.FileWriter(TENSORBOARD_TEST_LOGS_DIR, sess.graph)
    tfFileWriter.add_graph(sess.graph)
    tfFileWriter.close()

    # poslat k zapsani do Excelu vyplnene pole s vysledku testu pro vsechny snimky
    zapsatPolePoliJakoExcel(zahlaviKzapisuDoExcelu, vsechnyRadkyKzapisuDoExcelu, nazevExcelovskehoSouboru)

# end main

#######################################################################################################################
def prepare_file_system():


    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(TENSORBOARD_TEST_LOGS_DIR):
        # smazani predchozich logu - novy fungujici kod
        shutil.rmtree(TENSORBOARD_TEST_LOGS_DIR, ignore_errors=True)
        # smazani predchozich logu - puvodni kod z nejakeho duvodu nefungujici
        # tf.gfile.DeleteRecursively(TENSORBOARD_TRAINING_LOGS_DIR)
    # end if
    tf.gfile.MakeDirs(TENSORBOARD_TEST_LOGS_DIR)
    return
# end function

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TEST_INPUT_IMAGES_DIR):
        print('')
        print('ERROR: TEST_IMAGES_DIR "' + TEST_INPUT_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the test images?')
        print('')
        return False
    # end if

    if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
        print('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
        return False
    # end if

    if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
        print('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
        return False
    # end if

    return True
# end function

#######################################################################################################################
def writeResultOnImage(outputDirectoryName, openCVImage, timeStamp, fileName, resultText):
    # TODO: this function may take some further fine-tuning to show the text well given any possible image size

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    # choose a font
    fontFace = cv2.FONT_HERSHEY_DUPLEX

    # chose the font size and thickness as a fraction of the image size
    if (imageWidth > 500): # pro obrazky sirsi nez 500 px
        fontScale = imageWidth/1000 + 0.7  # se velikost fontu vypocita podle sirky
    else:  # jinak (pro obrazky mensi)
        fontScale = 1.0  # se pouzije vychozi velikost fontu
    #fontScale = 2.0

    fontThickness = 2

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area based on the text area center, width, and height
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    # write the text on the image
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, LABEL_FONT_COLOR, fontThickness, cv2.LINE_AA)

    # uložit výstupní obrázek
    cv2.imwrite(outputDirectoryName + "/" + timeStamp + "_test_output_" + fileName, openCVImage)
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()
