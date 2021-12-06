import csv
import sys
import gensim.downloader as api
import random

class wordModel:

    def __init__(self, modelName):
        print("initializing...")
        self.modelName = modelName
        self.C = 0
        self.V = 80
        self.readFile()
        self.loadCorpus()

    def readFile(self):
        print("reading synonyms...")
        self.file = open('synonyms.csv', 'r')
        self.csvreader = csv.reader(self.file)
        self.header = []
        self.header = next(self.csvreader)
        self.rows = []
        for row in self.csvreader:
            self.rows.append(row)
        self.file.close()
    
    def openOutput(self):
        self.originalOut = sys.stdout
        self.outputFile = open(self.modelName+'-details.csv', 'a')
        sys.stdout = self.outputFile

    def closeOutput(self):
        self.outputFile.close()
        sys.stdout = self.originalOut
    
    def loadCorpus(self):
        print("loading corpus...")
        self.corpus = api.load(self.modelName)
        self.model_size = len(self.corpus)

    def analyse(self):
        print("analyzing...")
        self.originalOut = sys.stdout
        self.analysisFile = open('analysis.csv', 'a')
        sys.stdout = self.analysisFile
        print(self.modelName + ", " + str(self.model_size) + ", " + str(self.C) + ", " + str(self.V) + ", " + str(self.C/self.V))
        self.analysisFile.close()
        sys.stdout = self.originalOut

    def operate(self):
        print("operating...")
        for row in self.rows:
            questionWord = row[0]
            answerWord = row[1]
            maximum = 0
            guessWord = ""
            labelString = ""
            if(not questionWord in self.corpus.key_to_index) or (not (row[2] in self.corpus.key_to_index or row[3] in self.corpus.key_to_index or row[4] in self.corpus.key_to_index or row[5] in self.corpus.key_to_index)):
                labelString = "guess"
                self.V -= 1
                guessWord = row[random.randint(2,5)]
            else:
                for i in range(2, 6):
                    if(row[i] in self.corpus.key_to_index):
                        cosine = self.corpus.similarity(questionWord, row[i])
                        if (cosine > maximum):
                            maximum = cosine
                            guessWord = row[i]
                if(answerWord == guessWord):
                    labelString = "correct"
                    self.C += 1
                else:
                    labelString = "wrong"
            self.openOutput()
            print(questionWord + ", " + answerWord + ", " + guessWord + ", " + labelString)
            self.closeOutput()
        self.analyse()

def main():
    print("Starting first model...")
    m1 = wordModel(modelName='word2vec-google-news-300')
    m1.operate()
    print("Finished first model.")

    print()

    print("Starting second model...")
    m2 = wordModel(modelName='fasttext-wiki-news-subwords-300')
    m2.operate()
    print("Finished second model.")

    print()

    print("Starting third model...")
    m3 = wordModel(modelName='glove-wiki-gigaword-300')
    m3.operate()
    print("Finished third model.")

    print()

    print("Starting fourth model...")
    m4 = wordModel(modelName='glove-twitter-100')
    m4.operate()
    print("Finished fourth model.")

    print()

    print("Starting fifth model...")
    m5 = wordModel(modelName='glove-twitter-200')
    m5.operate()
    print("Finished fifth model.")

if __name__ == "__main__":
	main()
