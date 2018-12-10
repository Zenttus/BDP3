#import tweets_catcher
#from tweets_catcher import HDFSManager

from text_classifiers import TextClassifierModel1, TextClassifierModel2

if __name__ == '__main__':

    #manager = HDFSManager()
    #tweets_catcher.get_tweets(manager)

    model1 = TextClassifierModel1('cleantextlabels7.csv')
    model2 = TextClassifierModel2('cleantextlabels7.csv')

    # Load weights
    # model.initiate_model('./sentiment_model1.h5')
    # model.initiate_mode2('./sentiment_model1.h5')
    # Create weights
    model1.initiate_model() #100 LSTM neurons
    model2.initiate_model() #10 LSTM neurons

    print(model1.predict('Love the movie.'))
    print(model1.predict('Hate the movie.'))
    print(model1.predict('movie was ok.'))

    print(model2.predict('Love the movie.'))
    print(model2.predict('Hate the movie.'))
    print(model2.predict('movie was ok.'))
