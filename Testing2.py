import streamlit as st
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.errors import HttpError
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as UC
from playwright.sync_api import sync_playwright
import pandas as pd
from io import StringIO
import joblib
import gdown
import time
import requests
import re
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


api_key = 'AIzaSyBmM-Z_PfxgXwOlnGNff4OzWCASjMIrpnw'
youtube = build('youtube', 'v3', developerKey=api_key)

model = joblib.load('sentiment_model.pkl')
#vectorizer = joblib.load('tfidf_vectorizer.pkl')

lemmatizer = WordNetLemmatizer()

file_id = "1iSkNMFXU5BXNNE9OyubvmSEsCFDRk-5w"
destination = "tfidf_vectorizer.pkl"

#download_url = f"https://drive.google.com/uc?id={file_id}"
download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

gdown.download(download_url, destination, quiet=False)

vectorizer = joblib.load(destination)


def get_channel_videos(channel_id, max_videos=15):
    videos = []
    next_page_token = None
    base_video_url = "https://www.youtube.com/watch?v="

    while len(videos) < max_videos:
        request = youtube.search().list(
            part='snippet',
            channelId=channel_id,
            maxResults=min(10, max_videos - len(videos)),  
            type='video',
            order='date',  
            pageToken=next_page_token 
        ).execute()

        video_ids = [item['id']['videoId'] for item in request.get('items', []) if 'videoId' in item['id']]
        if not video_ids:
            break  

        
        video_details_request = youtube.videos().list(
            part="contentDetails",
            id=",".join(video_ids)
        ).execute()

        for item, video_details in zip(request['items'], video_details_request['items']):
            if 'videoId' in item['id']:
                video_id = item['id']['videoId']
                video_title = item['snippet']['title']
                video_url = f"{base_video_url}{video_id}"

                
                duration = video_details['contentDetails']['duration']

                
                if 'M' not in duration or (duration.startswith('PT') and 'S' in duration and 'M' not in duration):
                    
                    continue

                videos.append({'video_id': video_id, 'video_title': video_title, 'video_url': video_url})

        next_page_token = request.get('nextPageToken')

        if not next_page_token:
            break

    return videos



def get_video_details(video_id):
    request = youtube.videos().list(part='statistics', id=video_id).execute()
    details = request['items'][0]['statistics']
    return details


def check_sponsorship_disclaimer(video_url):
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(video_url)

        try:
            reject_button = page.locator("//button[text()='Reject all']")
            reject_button.click(timeout=5000)
        except:
            pass  # If button is not found, ignore

        disclaimer = page.locator("//*[contains(text(), 'Includes paid promotion')]")
        if disclaimer.count() > 0:
            print("Sponsorship disclaimer found!")
            return True
        
        print("No sponsorship disclaimer found")
        browser.close()
        return False


def get_comments(video_id):
  
    comments = []
    
    try:
        
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100
        ).execute()

        
        for item in request['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        
        while 'nextPageToken' in request:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=request['nextPageToken']  
            ).execute()

            
            for item in request['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
        
        return comments

    except HttpError as e:
        
        if e.resp.status == 403 and 'commentsDisabled' in str(e):
            
            return None  
        else:
            
            raise

def extract_channel_id(youtube_url):

    youtube_url = youtube_url.strip()  

    
    match = re.match(r'(https?://)?(www\.)?youtube\.com/channel/([a-zA-Z0-9_-]+)', youtube_url)
    if match:
        return match.group(3)  

    
    match = re.match(r'(https?://)?(www\.)?youtube\.com/user/([a-zA-Z0-9_-]+)', youtube_url)
    if match:
        username = match.group(3)  
        return get_channel_id_by_username(username)

    
    match = re.match(r'(https?://)?(www\.)?youtube\.com/c/([a-zA-Z0-9_-]+)', youtube_url)
    if match:
        custom_name = match.group(3)  
        return get_channel_id_by_custom_name(custom_name)

    
    match = re.match(r'(https?://)?(www\.)?youtube\.com/([a-zA-Z0-9_-]+)', youtube_url)
    if match:
        homepage_name = match.group(3)
        return get_channel_id_by_custom_name(homepage_name)

    
    raise ValueError("Invalid YouTube URL format. Please enter a valid channel URL.")

def get_channel_id_by_username(username):
    
    request = youtube.channels().list(part='id', forUsername=username).execute()

    if request['items']:
        return request['items'][0]['id']
    else:
        raise ValueError(f"Could not find a channel for username: {username}")

def get_channel_id_by_custom_name(custom_name):
  
    request = youtube.search().list(part='snippet', q=custom_name, type='channel', maxResults=1).execute()

    if request['items']:
        return request['items'][0]['snippet']['channelId']
    else:
        raise ValueError(f"Could not find a channel for custom name: {custom_name}")


def analyze_sentiment(comments):
   
    if isinstance(comments, list):
        
        comment_array = vectorizer.transform(comments)
    else:
        
        comment_array = vectorizer.transform([comments])
    
    
    sentiment_scores = model.predict(comment_array)

    sentiment_mapping = {
        'positive': 1,
        'negative': 0,
        'neutral': 0.5 
    }
    
    
    numeric_sentiment_scores = np.array([sentiment_mapping.get(score, 0) for score in sentiment_scores], dtype=np.float64)
    
    
    return np.mean(numeric_sentiment_scores) if len(numeric_sentiment_scores) > 0 else 0


def is_video_sponsored(video_title, video_description, video_url):
   
    sponsored_keywords = [' is sponsored', 'paid promotion', 'partnered with', 'includes paid promotion', 'brand deal', 'paid partnership']
    
    combined_text = f"{video_title} {video_description}".lower()
    
    is_sponsored_by_keywords = any(keyword in combined_text for keyword in sponsored_keywords)

    is_sponsored_by_disclaimer = check_sponsorship_disclaimer(video_url)

    return is_sponsored_by_keywords or is_sponsored_by_disclaimer
    

def get_video_description(video_id):
  
    
    request = youtube.videos().list(
        part='snippet',  
        id=video_id
    ).execute()

    
    if request['items']:
        return request['items'][0]['snippet']['description']
    else:
        return None
    
def get_video_engagement_metrics(video_id):
  
    try:
        request = youtube.videos().list(
            part="statistics",
            id=video_id
        ).execute()

        if 'items' in request and len(request['items']) > 0:
            stats = request['items'][0]['statistics']
            likes = int(stats.get('likeCount', 0))
            views = int(stats.get('viewCount', 0))
            comments_Count = int(stats.get('commentCount', 0))
            
            return likes, views, comments_Count
        else:
            return 0, 0, 0  
    except Exception as e:
        st.error(f"Error fetching engagement metrics for video {video_id}: {str(e)}")
        return 0, 0, 0



def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stopwords.words('english')]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)



def get_channel_name(channel_id):
  
    request = youtube.channels().list(part='snippet', id=channel_id).execute()
    if request['items']:
        return request['items'][0]['snippet']['title']
    else:
        return None


#----------------------------------------------------------------------------------------------------

def evaluate_channel(channel_id):
  
    videos = get_channel_videos(channel_id)  
    sponsored_sentiments = []
    unsponsored_sentiments = []
    sponsored_engagement_metrics = {"likes": 0, "views": 0, "comments": 0, "count": 0}
    unsponsored_engagement_metrics = {"likes": 0, "views": 0, "comments": 0, "count": 0}
    sponsored_videos = []
    unsponsored_videos = []

    for video in videos:
        video_id = video['video_id']
        video_title = video['video_title']
        video_url = video['video_url']
        video_description = get_video_description(video_id)
        

        
        #details = get_video_details(video_id)
        #comments = get_comments(video_id)
        
        #comments = comments.apply(clean_text)
        #comments = [clean_text(comment) for comment in comments]

        
        #sentiment_score = analyze_sentiment(comments)
        
        
        #if 'sponsored' in video_title.lower() or 'ad' in video_title.lower():
            #sponsored_sentiments.append(sentiment_score)
        #else:
            #unsponsored_sentiments.append(sentiment_score)

        comments = get_comments(video_id)    

        if comments is None:
            continue  

        comments = [clean_text(comment) for comment in comments]

        likes, views, comments_Count = get_video_engagement_metrics(video_id)   


        if is_video_sponsored(video_title, video_description, video_url):
            
            sponsored_sentiments.append(analyze_sentiment(comments))

            sponsored_videos.append(video)

            sponsored_engagement_metrics['likes'] += likes
            sponsored_engagement_metrics['views'] += views
            sponsored_engagement_metrics['comments'] += comments_Count
            sponsored_engagement_metrics['count'] += 1

        else:

            unsponsored_sentiments.append(analyze_sentiment(comments))

            unsponsored_videos.append(video)

            unsponsored_engagement_metrics['likes'] += likes
            unsponsored_engagement_metrics['views'] += views
            unsponsored_engagement_metrics['comments'] += comments_Count
            unsponsored_engagement_metrics['count'] += 1


    num_sponsored = len(sponsored_videos)
    num_unsponsored = len(unsponsored_videos)   

    #st.write(f"Total Sponsored Videos: {num_sponsored}")
    #st.write(f"Total Unsponsored Videos: {num_unsponsored}")
    #st.write(f"Total Sponsored Sentiment Videos: {len(sponsored_sentiments)}")
    #st.write(f"Total Unsponsored Sentiment Videos: {len(unsponsored_sentiments)}")
    
   
    avg_sponsored_sentiment = np.mean(sponsored_sentiments) if sponsored_sentiments else 0
    avg_unsponsored_sentiment = np.mean(unsponsored_sentiments) if unsponsored_sentiments else 0

    if sponsored_engagement_metrics['views'] > 0:
        avg_sponsored_engagement_score = (sponsored_engagement_metrics['likes'] + sponsored_engagement_metrics['comments']) / sponsored_engagement_metrics['views']
    else:
        avg_sponsored_engagement_score = 0

    
    if unsponsored_engagement_metrics['views'] > 0:
        avg_unsponsored_engagement_score = (unsponsored_engagement_metrics['likes'] + unsponsored_engagement_metrics['comments']) / unsponsored_engagement_metrics['views']
    else:
        avg_unsponsored_engagement_score = 0


    #avg_sponsored_engagement_score = np.mean(sponsored_engagement_metrics) if sponsored_engagement_metrics else 0
    #avg_unsponsored_engagement_score = np.mean(unsponsored_engagement_metrics) if unsponsored_engagement_metrics else 0
  

    #avg_sentiment_score = (avg_sponsored_sentiment + avg_unsponsored_sentiment) / 2

    overall_sponsored_score = (0.5 * avg_sponsored_sentiment) + (0.5 *  avg_sponsored_engagement_score)
    overall_unsponsored_score = (0.5 * avg_unsponsored_sentiment) + (0.5 *  avg_unsponsored_engagement_score)


    return overall_sponsored_score, overall_unsponsored_score, num_sponsored, num_unsponsored


st.title("YouTube Partner Estimator Tool")

st.subheader("Input Channel URL")
channel_url = st.text_input("Enter YouTube Channel URL")

if st.button("Evaluate"):

    channel_id = extract_channel_id(channel_url)
    channel_name = get_channel_name(channel_id)

    #avg_sponsored_sentiment, avg_unsponsored_sentiment = evaluate_channel(channel_id)
    overall_sponsored_score, overall_unsponsored_score, num_sponsored, num_unsponsored  = evaluate_channel(channel_id)

    #st.write(f"Average Sponsored Sentiment Score: {avg_sponsored_sentiment}")
    #st.write(f"Average Unsponsored Sentiment Score: {avg_unsponsored_sentiment}")

    st.subheader("Results")
    #st.write(f"YouTuber Name: {channel_name}")
    #st.write(f"Average Sponsored Content Score: {overall_sponsored_score}")
    #st.write(f"Average Unsponsored Content Score: {overall_unsponsored_score}")
    
    # After evaluating the channel, you can display the results in a table
    results = pd.DataFrame({
        'A': ['Overall Sponsored Score', 'Overall Unsponsored Score', 'YouTuber Name','Channel ID','Total Sponsored Videos','Total Unsponsored Videos'],
        'B': [overall_sponsored_score, overall_unsponsored_score, channel_name, channel_id, num_sponsored, num_unsponsored]

    })

    st.dataframe(results)


    if overall_sponsored_score > overall_unsponsored_score:
        st.success("This YouTuber is a potential good partner for sponsored videos!")
    else:
        st.warning("This YouTuber may not be a good fit for sponsored videos.")


