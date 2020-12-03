#!/usr/bin/python
import time

import pandas as pd
import tweepy

file_name = "twitter.txt"

API_KEY = 'e5JzKA0IvN7Tt3r5rliiVUqzk'
API_KEY_SECRET = 'cl1aI32dl2A7alJh2QN46pFScn7BBNGNtcvGULmDApwPKBvO1r'
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAANUYHQEAAAAApfso%2F8OYpqdMau6hcqctyWlFBts%3DNogpscSTTD7X5OBypqkIOehQ7I2SUdF2UjHXWj2GEln6B0dKhY'
ACCESS_TOKEN = '913728157907165184-2U5hki7BOYYon78psGgWFgjxs7QgPsh'
ACCESS_SECRET = 'ERUXbrCtn4IxQXGOQROlJCh1KlfRb81hK1bx3dooxwUOD'

auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

dataset = []
sleepTime = 2
api.wait_on_rate_limit = True
api.wait_on_rate_limit_notify = True

for line in open(file_name):
    time.sleep(sleepTime)

    fields = line.rstrip().split('\t')
    tweetid = fields[0]
    userid = fields[1]
    # print(fields, tweetid, userid)
    try:
        tweetFetched = api.get_status(tweetid)
        # print("Tweet fetched" + tweetFetched.text)
        print(len(dataset), tweetFetched.text)
        text = tweetFetched.text
        dataset.append([tweetid, text, tweetFetched.created_at, tweetFetched])
    except Exception as e:
        print("Inside the exception - ", tweetid, e)
        continue

df = pd.DataFrame(dataset)
df.to_csv('dataset.csv', index=False, header=False)
