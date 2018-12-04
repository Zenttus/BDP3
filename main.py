#import tweets_catcher
#from tweets_catcher import HDFSManager

from text_classifiers import TextClassifierModel1

if __name__ == '__main__':

    #TODO spark analysis
    #TODO gui

    #manager = HDFSManager()
    #tweets_catcher.get_tweets(manager)

    model = TextClassifierModel1('cleantextlabels7.csv')
    model.initiate_model()
