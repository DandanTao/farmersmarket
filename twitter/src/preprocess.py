import cleanup_tweet
import merge_sentences
import separate_sentences
import merge_to_df

def main():
    cleanup_tweet.process_all()
    # merge_sentences.merge_all()
    # separate_sentences.separate_sentences()
    # merge_to_df.merge_to_df()

if __name__ == '__main__':
    main()
