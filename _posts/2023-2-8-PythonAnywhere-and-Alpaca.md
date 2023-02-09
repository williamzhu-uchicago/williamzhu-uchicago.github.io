---
layout: post
title: Tapping into Algorithmic Trading with Alpaca API and PythonAnywhere
---

Recently I started exploring ways to deploy my very first algorithmic trading streategy. My tech-stack is extremely simple: trade with *Alpaca-py*, and deploy to cloud with *PythonAnywhere*. So far I am very happy with Alpaca for its intuitive API, fast account approval, and paper trading functionality. PythonAnywhere's price tag of \$5 per month is unbeatable as well.

The whole process was much easier than I had thought: I generated the API keys on Alpaca, wrote the Python script, upload the file onto PythonAnywhere, downloaded/updated the necessary libraries on the platform via *Bash*, and I am done. The most challenging part was to deal with the timing of my strategy using `time.sleep()` and `datetime.datetime.now()`. Eventually, I decided to write the Python script such that it will offer me a greater degree of flexibility by seperating the strategy part of the code from other funtionalities. Doing so can help me deploy strategies much faster since I will only have to change a specific and contained part of my code (unless I want to trade cryto). 

And I decided to share it here:


```python
import datetime
import time

from pytz import timezone
tz = timezone('EST')

import logging
logging.Formatter.converter = lambda *args: datetime.datetime.now(tz=tz).timetuple()
FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='./strategy.log', format=FORMAT, datefmt="%Y-%m-%d %H:%M:%S", \
    level=logging.INFO)
logging.info('Logging started')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

trading_client = TradingClient('<API_key>', '<API_secret_key>', paper=True)

def strategy():
    """
    Write the strategy here.
    Input: us_ticker (str), size(float): size of order
    """
    ...

def time_to_open(curr_time):
    """
    Helper function that returns the time from now to next day's market open.
    Input: curr_time (datetime): current time
    Output: time till market open (timedelta)
    """
    if curr_time.weekday() <= 4:
        if datetime.time(0, 0) < datetime.datetime.now(tz).time() < datetime.time(9, 30):
            date_to_open = curr_time.date()
        else:
            date_to_open = (curr_time + datetime.timedelta(days=1)).date()
    else:
        date_to_open = (curr_time + datetime.timedelta(days= 7 - curr_time.weekday())).date()
    next_open_time = datetime.datetime.combine(date_to_open, datetime.time(9, 30, tzinfo=tz))
    return next_open_time - curr_time

def run_algo():
    """
    Main function that connect all things together
    """
    print('Algorithm started')
    logging.info("Algorithm started")
    while True:
        if 0 <= datetime.datetime.now(tz).weekday() <= 4 and \
            datetime.time(9, 30) < datetime.datetime.now(tz).time() <= datetime.time(16, 0):
            print("Market opens. Executing strategy...")
            logging.info("Market opens. Executing straegy")
            strategy()
        else: # If not trading day, sleep till market open
            print("Market closed. Entering Sleep.")
            sleep_time = time_to_open(datetime.datetime.now(tz))
            logging.info("Market closed, sleeping for: %s", str(sleep_time))
            time.sleep(sleep_time.total_seconds() // 1 + 0.5)
```

I did these with the help from [this post](https://alpaca.markets/learn/pythonanywhere-trading-algorithm/).
