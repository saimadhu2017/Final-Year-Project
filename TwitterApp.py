import datashader.transfer_functions as tf
import datashader as ds
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pylab as pl
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

consumerKey = "LXOOgp0ol34ej0ODvvJlDGJ0J"
consumerSecret = "LetsfYng9oCwjbNzHpVsEEQ76M65L2YtiQbpd9toxDXCuJxDRT"
accessToken = "1297574824890396673-GSgXcGidDANj6ciOX5kwbzoZIVLFMs"
accessTokenSecret = "QTFIe7aLmB7wmksQGLXS9AAoq4qhLAz7GBEpG9TRSos5G"


# Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)
# Creating the API object while passing in auth information
api = tweepy.API(authenticate, wait_on_rate_limit=True)


def app():
    # Decorating the Nav Bar which is left side of APP
    st.sidebar.header("NAVIGATION")
    st.title("REAL TIME SENTIMENT ANALYSIS ON TWITTER ✨")
    activities = ["Collect The Data", "Clean The Data", "Summary View of Tweets Collected",
                  "Tweet Analyzer Using K Means", "Tweet Analyser with search keyword"]

    # choice = st.sidebar.selectbox("Type Of your Activity", activities)
    choice = st.sidebar.radio("Select Your Activity", activities)
    st.sidebar.header("CONFIGURATION NAV BAR")
    No_Of_Tweets = st.sidebar.slider("No of Tweets", 100, 5000)
    No_Of_Tweets_In_String = str(No_Of_Tweets)
    UserID = st.text_area(
        "Enter the exact twitter User Id of the User you want to Analyze (without @)")
    st.markdown(
        "🔴 Note--> Don't move to next Navbar Section without completing this section.")
    TweetsData = pd.DataFrame()

    def collectData():
        posts = tweepy.Cursor(api.user_timeline, screen_name=UserID,
                              tweet_mode="extended").items(No_Of_Tweets)
        TweetsData = pd.DataFrame(
            [tweet.full_text for tweet in posts], columns=['Tweets'])
        return(TweetsData)

    # Step 1
    if(choice == "Collect The Data"):
        st.subheader("Analyze the tweets of your favourite User 👦👧:")
        st.subheader("This tool performs the following tasks as given below:")
        st.write("1. Fetches the 5 most recent tweets from the given twitter handel")
        st.write("2. It also collects the tweets which are tweeted by a userID")
        st.write("3. In this section you just collect the Real Time data and can be used for analyses purpose in next section.")

        if(st.button("Collect The Data")):
            st.success("Data is being Collected wait for some time")
            posts = tweepy.Cursor(
                api.user_timeline, screen_name=UserID, tweet_mode="extended").items(No_Of_Tweets)
            TweetsData = pd.DataFrame(
                [tweet.full_text for tweet in posts], columns=['Tweets'])
            st.write(TweetsData)

    # Step 2
    elif(choice == "Clean The Data"):
        TweetsData = collectData()

        def cleanTxt(text):
            # Removing @mentions
            text = re.sub('@[A-Za-z0–9]+', '', text)
            text = re.sub('#', '', text)  # Removing '#' hash tag
            text = re.sub('RT[\s]+', '', text)  # Removing RT
            text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
            return(text)

        # Clean the tweets
        TweetsData['Tweets'] = TweetsData['Tweets'].apply(cleanTxt)
        st.write(TweetsData)

    # Step 3
    elif(choice == "Summary View of Tweets Collected"):
        TweetsData = collectData()

        def cleanTxt(text):
            # Removing @mentions
            text = re.sub('@[A-Za-z0–9]+', '', text)
            text = re.sub('#', '', text)  # Removing '#' hash tag
            text = re.sub('RT[\s]+', '', text)  # Removing RT
            text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
            return(text)
        # Clean the tweets
        TweetsData['Tweets'] = TweetsData['Tweets'].apply(cleanTxt)

        st.subheader("Analyze the tweets of your favourite User 👦👧:")
        st.subheader("This tool performs the following tasks as given below:")
        st.write("1. Fetches the 5 most recent tweets from the given twitter handel")
        st.write("2. Generates a Word Cloud")
        st.write(
            "3. Performs Sentiment Analysis a displays it in form of a Bar Graph")

        Analyzer_choice = st.selectbox("Select the Activities",  [
                                       "Show Recent Tweets", "Generate WordCloud", "Visualize the Sentiment Analysis"])

        if(st.button("Analyze")):
            if(Analyzer_choice == "Show Recent Tweets"):
                st.success("Fetching last 5 Tweets")

                def Show_Recent_Tweets(raw_text):
                    rl = []
                    for i in range(0, 6):
                        rl.append(TweetsData['Tweets'][i])
                    return(rl)

                recent_tweets = Show_Recent_Tweets(UserID)
                st.write(recent_tweets)

            elif(Analyzer_choice == "Generate WordCloud"):
                st.success("Generating Word Cloud")

                def gen_wordcloud():
                    df = TweetsData
                    # word cloud visualization
                    allWords = ' '.join([twts for twts in df['Tweets']])
                    wordCloud = WordCloud(
                        width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
                    plt.imshow(wordCloud, interpolation="bilinear")
                    plt.axis('off')
                    plt.savefig('WC.jpg')
                    img = Image.open("WC.jpg")
                    return(img)

                img = gen_wordcloud()
                st.image(img)

            else:
                def Plot_Analysis():
                    st.success(
                        "Generating Visualisation for Sentiment Analysis of the User Given.")
                    df = TweetsData

                    # Create a function to clean the tweets
                    def cleanTxt(text):
                        # Removing @mentions
                        text = re.sub('@[A-Za-z0–9]+', '', text)
                        text = re.sub('#', '', text)  # Removing '#' hash tag
                        text = re.sub('RT[\s]+', '', text)  # Removing RT
                        # Removing hyperlink
                        text = re.sub('https?:\/\/\S+', '', text)
                        return(text)

                    # Clean the tweets
                    df['Tweets'] = df['Tweets'].apply(cleanTxt)

                    # Create a function to get the subjectivity
                    def getSubjectivity(text):
                        return(TextBlob(text).sentiment.subjectivity)

                        # Create a function to get the polarity
                    def getPolarity(text):
                        return(TextBlob(text).sentiment.polarity)

                    # Create two new columns 'Subjectivity' & 'Polarity'
                    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
                    df['Polarity'] = df['Tweets'].apply(getPolarity)

                    def getAnalysis(score):
                        if(score < 0):
                            return("Negative")
                        elif(score == 0):
                            return("Netural")
                        else:
                            return("Positive")

                    df['Analysis'] = df['Polarity'].apply(getAnalysis)
                    return(df)

                df = Plot_Analysis()
                st.write(sns.countplot(x=df["Analysis"], data=df))
                st.pyplot(use_container_width=True)
    elif(choice == "Tweet Analyzer Using K Means"):
        st.subheader(
            "This tool fetches the last "+No_Of_Tweets_In_String+" tweets from the twitter handel & Performs the following tasks")
        st.write("1. Converts it into a DataFrame")
        st.write("2. Cleans the text")
        st.write(
            "3. Analyzes Subjectivity of tweets and adds an additional column for it")
        st.write(
            "4. Analyzes Polarity of tweets and adds an additional column for it")
        st.write(
            "5. Analyzes Sentiments of tweets and adds an additional column for it")

        if(st.button("Analyze")):
            st.success("Please wait while working on K means......")
            TweetsData = collectData()
            st.success(
                "Collecting the data once again to verify we got the right data or not...")

            def cleanTxt(text):
                # Removing @mentions
                text = re.sub('@[A-Za-z0–9]+', '', text)
                text = re.sub('#', '', text)  # Removing '#' hash tag
                text = re.sub('RT[\s]+', '', text)  # Removing RT
                text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
                return(text)
            # Clean the tweets
            st.success("Cleaning the data ...")
            TweetsData['Tweets'] = TweetsData['Tweets'].apply(cleanTxt)

            df = TweetsData
            # TFIDF Approach
            tf_idf_vect = CountVectorizer(analyzer='word', ngram_range=(
                1, 1), stop_words='english', min_df=0.0001)
            tf_idf_vect.fit(df['Tweets'])  # Here the formula internally works
            desc_matrix = tf_idf_vect.transform(df["Tweets"])
            st.success("Done with TFIDF Approach ...")
            st.success("working on K means...")
            # K Means Clustering
            num_clusters = 3
            km = KMeans(n_clusters=num_clusters)
            km.fit(desc_matrix)
            clusters = km.labels_.tolist()

            st.success("Done with Clustering see the below results...")

            # create DataFrame films from all of the input files.
            st.subheader(
                "KMeans Clustring which have 3 clusters 1,0 and -1 as values:")
            tweets = {'Tweet': df["Tweets"].tolist(), 'Cluster': clusters}
            frame = pd.DataFrame(tweets, index=[clusters])
            st.write(frame)

            st.subheader(
                "Showing the Different Clusters which are +ve, -ve and Neutral:")
            st.write("Positive Cluster:")
            st.write(frame[frame['Cluster'] == 1])  # positive cluster
            st.write("Neutral cluster:")
            st.write(frame[frame['Cluster'] == 2])  # Neutral cluster
            st.write("Negative cluster:")
            st.write(frame[frame['Cluster'] == 0])  # Negative cluster
            st.write("Final summary:")
            # neutral =
            st.write(frame['Cluster'].value_counts())

    else:
        st.subheader(
            "This tool fetches the last "+No_Of_Tweets_In_String+" tweets from the twitter handel & Performs the following tasks")
        st.write("1. Converts it into a DataFrame")
        st.write("2. Cleans the text")
        st.write(
            "3. Analyzes Subjectivity of tweets and adds an additional column for it")
        st.write(
            "4. Analyzes Polarity of tweets and adds an additional column for it")
        st.write(
            "5. Analyzes Sentiments of tweets and adds an additional column for it")

        SearchKeyword = st.text_area(
            "*Enter the keyword to be search to collect the data*")
        st.markdown(
            "<--------Also Do checkout the another cool tool from the sidebar")
        date_since = "2010-1-1"

        def get_data(SearchKeyword):
            st.success("Fetching Last "+No_Of_Tweets_In_String +
                       " Tweets Please wait........")
            posts = tweepy.Cursor(
                api.search, q=SearchKeyword, lang="en", since=date_since).items(No_Of_Tweets)
            st.success("Collected the data and converting it to Data frame.")
            df = pd.DataFrame(
                [tweet.text for tweet in posts], columns=['Tweets'])
            st.success("converted to data frame of collected Tweets")

            # cleaning the twitter text function
            def cleanTxt(text):
                text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
                text = re.sub('#', '', text)  # Removing '#' hash tag
                text = re.sub('RT[\s]+', '', text)  # Removing RT
                text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
                return(text)

            # Clean the tweets
            df['Tweets'] = df['Tweets'].apply(cleanTxt)
            st.success("Finished with cleaning the data")

            def getSubjectivity(text):
                return(TextBlob(text).sentiment.subjectivity)

            # Create a function to get the polarity
            def getPolarity(text):
                return(TextBlob(text).sentiment.polarity)

            # Create two new columns 'Subjectivity' & 'Polarity'
            df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
            df['Polarity'] = df['Tweets'].apply(getPolarity)
            st.success(
                "Assainged the values of subjectivity and polarity for each tweet.")

            def getAnalysis(score):
                if(score < 0):
                    return("Negative")
                elif(score == 0):
                    return("Neutral")
                else:
                    return("Positive")
            st.success("Final step please wait.........")
            df['Analysis'] = df['Polarity'].apply(getAnalysis)
            return(df)

        if(st.button("Show Data")):
            df = get_data(SearchKeyword)
            st.write(df)
    st.subheader(
        ' ---------------Created By :  Project 304 @KL University --------------- :sunglasses:')


if __name__ == "__main__":
    app()
