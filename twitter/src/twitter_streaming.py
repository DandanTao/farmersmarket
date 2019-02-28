from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import os

access_token = '815767018930667524-gW8GyNBnADk3hbHo3Zt3prgAIaxtLY3'
access_token_secret = 'wPUhjlQU1sn81MTdLKIrNbOR6FMhSMbx0dwevsBj30j2Z'
consumer_key = 'SzgwcufmBOui6cSrT27AgcFdt' # API Key
consumer_secret = 'MdYJ88qDUYb3CGagx5mJ81Hg06Y6VzpHxiSdPbujEAw5ilUtn0' # API Secret Key

FILE_PATH = "../JSON_data/twitter_data_2132019.json"

class MyListener(StreamListener):

    def on_data(self, data):
        try:
            with open(FILE_PATH, 'a') as f:
                f.write(data)
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return status

def main():

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    print("Streaming tweets now ...")
    stream = Stream(auth, MyListener(), tweet_mode = 'extended')
    stream.filter(track=['farmers market', 'farmersmarket', '#farmers market', '#FarmersMarket'])
    return

if __name__ == '__main__':
    main()
