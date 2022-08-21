from unittest.mock import sentinel
from flask import Flask, request, jsonify
import tweepy
from tweepy.models import Status
from os import getenv
from transformers import pipeline

MIN_POKEMON_ID = 1
MAX_POKEMON_ID = 111

bearer = getenv(
    "bearer",
    "AAAAAAAAAAAAAAAAAAAAAKzbgAEAAAAAYK00KnsHAEDgqL4ARmF6Rq%2B0AVA%3D5NycDrS2tOhUClJyVoOZLLmt2XGtkvOeW4ngpMLWs66Q4chQmC"
)

auth = tweepy.OAuth2BearerHandler(bearer)
api = tweepy.API(auth)

app = Flask(__name__)


model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline(
    "sentiment-analysis",
    model=model_path,
    tokenizer=model_path
)


@app.route("/", methods=['POST'])
def hello_world():
    data = request.json
    tweet_id = data['url'].split("/")[-1]
    x: Status = api.get_status(tweet_id, tweet_mode='extended')
    sentiment = sentiment_task([x.full_text])
    pokemon_id = sentiment2pokemon_id(sentiment)
    return jsonify({'text': x.full_text, 'pokemon_id': pokemon_id})


def sentiment2pokemon_id(sentiment):
    s = sentiment[0]
    label: str = s['label']
    score: float = s['score']
    score = score * - 1 if label == 'Negative' else score
    return normalize(score)


def normalize(score):
    width = MAX_POKEMON_ID - MIN_POKEMON_ID
    return int((score - (-1))/(1 - (-1)) * width + MIN_POKEMON_ID)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
