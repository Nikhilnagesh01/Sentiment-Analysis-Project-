import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import time
import warnings
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from collections import Counter

# Ignore warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Download NLTK resources (uncomment if running for the first time)
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')
# nltk.download('punkt')

# Function to scrape reviews from Amazon
@st.cache_data
def get_reviews(url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    
    driver.get(url)
    reviews = []

    while True:
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-hook="review"]'))
            )
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            review_divs = soup.select('div[data-hook="review"]')

            for review in review_divs:
                review_text = review.select_one('span[data-hook="review-body"]').get_text(strip=True)
                reviews.append(review_text)

            try:
                next_page = driver.find_element(By.CSS_SELECTOR, 'li.a-last a')
                next_page.click()
                time.sleep(2)  # Add a small delay to allow the next page to load
            except NoSuchElementException:
                st.info("No more pages left to scrape.")
                break  # No more pages left
        except TimeoutException:
            st.warning("Loading took too much time! Check your connection or the URL.")
            break
        except Exception as e:
            st.error(f"An error occurred: {e}")
            break

    driver.quit()
    return reviews

# Function to preprocess text (cleaning, tokenization, lemmatization)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = word_tokenize(text)
    stopwords_set = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stopwords_set]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to extract nouns using NLTK
def extract_nouns(text):
    tokens = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(tokens)
    nouns = [word for word, tag in tagged_words if tag.startswith('NN')]
    return nouns

# Function to analyze sentiment using NLTK Vader
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

# Main function to run Streamlit app
def main():
    st.title("Amazon Reviews Analysis")

    # Input field for Amazon product URL
    amazon_url = st.text_input("Enter Amazon Product URL:")

    if amazon_url:
        # Scrape reviews
        st.subheader("Scraping reviews...")
        all_reviews = get_reviews(amazon_url)
        if all_reviews:
            df = pd.DataFrame(all_reviews, columns=['Review'])

            # Clean reviews
            df['Cleaned_Review'] = df['Review'].apply(preprocess_text)

            # Display review length distribution
            st.subheader('Review Length Distribution')
            fig, ax = plt.subplots()
            sns.histplot(df['Cleaned_Review'].apply(len), bins=30, kde=True, color='skyblue', ax=ax)
            ax.set_title('Review Length Distribution')
            ax.set_xlabel('Review Length')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

            # Display word cloud of most common words in reviews
            st.subheader('Word Cloud of Most Common Words in Reviews')
            fig, ax = plt.subplots()
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Cleaned_Review']))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Top Words in Reviews')
            st.pyplot(fig)

            # Extract nouns and display top 10 nouns in reviews
            df['Nouns'] = df['Cleaned_Review'].apply(extract_nouns)
            all_nouns = [noun for review_nouns in df['Nouns'] for noun in review_nouns]
            top_nouns = Counter(all_nouns).most_common(10)

            st.subheader('Top 10 Nouns in Reviews')
            fig, ax = plt.subplots()
            ax.barh([word for word, _ in top_nouns], [count for _, count in top_nouns], color='skyblue')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Nouns')
            ax.set_title('Top 10 Nouns in Reviews')
            ax.invert_yaxis()
            st.pyplot(fig)

            # Analyze sentiment and display sentiment analysis plot
            df['Sentiment'] = df['Cleaned_Review'].apply(analyze_sentiment)
            df['Sentiment_Type'] = df['Sentiment'].apply(lambda x: 'Positive' if x['compound'] > 0 else ('Negative' if x['compound'] < 0 else 'Neutral'))

            st.subheader('Sentiment Analysis of Amazon Reviews')
            fig, ax = plt.subplots()
            sns.countplot(x='Sentiment_Type', data=df, palette='Set2', ax=ax)
            ax.set_title('Sentiment Analysis of Amazon Reviews')
            st.pyplot(fig)

            # Display overall word cloud of cleaned reviews
            st.subheader('Word Cloud of Amazon Reviews')
            fig, ax = plt.subplots()
            wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Cleaned_Review']))
            ax.imshow(wordcloud_all, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud of Amazon Reviews')
            st.pyplot(fig)

            # Save dataframe to CSV
            df.to_csv('amazon_reviews.csv', index=False)
            st.markdown("Scraping completed and saved to amazon_reviews.csv")
        else:
            st.warning("No reviews were scraped. Please check the URL or try again later.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
