---
layout: post
title: Yves Hilpisch - Python for Algorithmic Trading - Chapter 6
---

# Building Classes for Event-Based Backtesting

**Vectorized backtesting** is fast to execute but cannot cope with all types of trading strategies. Potential shortcomings of the approach include:
1. Look-ahead bias: backtesting is based on a complete data set without taking into accoutn new data arrivals
2. Simplification: fixed transaction costs cannot be modeled by vectorization, and the non-divisibility of stocks cannot be modeled properly
3. Non-recursiveness

**Event-based backtesting** addresses these issue. An *event* is defined as the arrival of new data. A new event is generally identified by a *bar*. which represents one unit of new data (e.g. one-minute bar implies event occurs per minute). Advantages of event-based backtesting include:
1. Incremental approach: backtesting takes place on the premise that new data arrives incrementally
2. Path dependency: able to keep track of conditional, recursive, and path-dependent statistics
3. Reusability: scripts written for one strategy can be used for another
4. Automation-ready

## Backtesting Base Class
A Python class supporting event-based backtesting should include the following functionalities:
1. Data retrieving and clearning support
2. Helper and convenience functions
3. Order-placing functionality (for both buy and sell)
4. Position-closing functionality


```python
import numpy as np
import pandas as pd
from pylab import plt
```


```python
class BacktestBase:
    def __init__(self, symbol, start, end, amount, ftc=0.0, ptc=0.0, verbose=True):
        self.symbol = symbol
        self.start, self.end = start, end #starting and ending period
        self.initial_amount = amount #Stores initial principal, to be left unchanged
        self.amount = amount #Starting cash balance that can be changed
        self.ftc, self.ptc = ftc, ptc #fixed and proportional transaction costs per trade
        self.units = 0 #initial unit of instrument (e.g. shares)
        self.position = 0 #Initial position set to neutral
        self.trades = 0 #Number of trades
        self.verbose = verbose #Verbose mode allows full report for later use
        self.get_data() #See below

    def get_data(self): #Fulfills functionality 1 above
        """
        Retrieves and cleans the data.
        """
        raw = pd.read_csv("Data/pyalgo_eikon_eod_data.csv", index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["log_pc"] = np.log(raw["price"] / raw["price"].shift(1))
        self.data = raw.dropna()
```


```python
class BacktestBase(BacktestBase): #Class inheritence
    def plot_data(self, cols="price"): #Part of functionality 2
        """
        Plots specified columns of dataframe.
        
        Input: cols(str or list of str): columns to be plotted
        """
        self.data[cols].plot(title=self.symbol)

    def get_date_price(self, bar):
        """
        For a specific bar(datetime), return the date and price of that bar.

        Input: bar(str in date format)
        """
        date = str(self.data.index[bar])[:10]
        price = self.data.price.iloc[bar]
        return date, price

    def print_balance(self, bar):
        """
        Print out cash balance info given a bar

        Input: bar(str in date format)
        """
        date, price = self.get_date_price(bar)
        print(f'{date} | current balance {self.amount:.2f}')

    def print_net_wealth(self, bar):
        """
        Print out current cash balance info given a bar

        Input: bar(str in date format)
        """
        date, price = self.get_date_price(bar)
        net_wealth = self.units * price + self.amount
        print(f'{date} | current net wealth {net_wealth:.2f}')
```

With the class established, we can test its functionalities:


```python
bb = BacktestBase(symbol="AAPL.O", start="2010-1-1", end="2019-12-31", amount=10000)
bb.plot_data()
```


```python
class BacktestBase(BacktestBase): #Part of functionality 3
    def place_buy_order(self, bar, units=None, amount=0):
        """
        Place a buy order.
        
        Inputs:
            bar (str in date format)
            units (int): number of instrument to buy
            amount (int): numerical value of instrument to buy
        """
        date, price = self.get_date_price(bar)
        if units is None:
            units = amount // price
        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        if self.verbose:
            print(f'{date} | buying {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def place_sell_order(self, bar, units=None, amount=0):
        """
        Place a sell order.
        
        Inputs:
            bar (str in date format)
            units (int): number of instrument to sell
            amount (int): numerical value of instrument to sell
        """
        date, price = self.get_date_price(bar)
        if units is None:
            units = amount // price
        self.amount += (units * price) * (1 - self.ptc) - self.ftc
        self.units -= units
        self.trades += 1
        if self.verbose:
            print(f'{date} | selling {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)
```


```python
class BacktestBase(BacktestBase): #Part of functionality 4
    def close_out(self, bar):
        """
        Closing out a long or short position.
        """
        date, price = self.get_date_price(bar)
        self.amount += self.units * price
        self.units = 0
        self.trades += 1
        if self.verbose:
            print(f'{date} | inventory {self.units} units at {price:.2f}')
            print('=' * 55)
        print('Final balance [$] {:.2f}'.format(self.amount))
        perf = ((self.amount - self.initial_amount) /self.initial_amount * 100)
        print('Net Performance [%] {:.2f}'.format(perf))
        print('Trades Executed [#] {:.2f}'.format(self.trades))
        print('=' * 55)
```

Now, we can easily implement a L/S mean reversion strategy backtest by adding more methods into the BacktestBase class


```python
class BacktestBase(BacktestBase): #Class inheritence

    def go_long(self, bar, units=None, amount=0):
        """
        Go long position.
        
        Inputs:
            bar (str in date format)
            units (int): number of instruments to buy
            amount (int/"all"): numerical value of instruments to buy
        """
        if self.position == -1: #If we are in a short position, go neutral
            self.place_buy_order(bar, units=-self.units)
        if units:
            self.place_buy_order(bar, units=units)
        elif amount != 0:
            if amount == 'all':
                amount = self.amount
            self.place_buy_order(bar, amount=amount)

    def go_short(self, bar, units=None, amount=0):
        """
        Same as above but the other direction
        """
        if self.position == 1:
            self.place_sell_order(bar, units=self.units)
        if units:
            self.place_sell_order(bar, units=units)
        elif amount != 0:
            if amount == 'all':
                amount = self.amount
            self.place_sell_order(bar, amount=amount)
            
    def run_mr_strategy(self, SMA, threshold):
        """
        Backtests a mean reversion strategy.
        
        Inputs:
            SMA (int): simple moving average window in days
            threshold (float): value for deviation-based signal relative to SMA
        """
        msg = f'\n\nRunning mean reversion strategy | SMA={SMA} & thr={threshold}'
        msg += f'\nfixed costs={self.ftc} | proportional costs={self.ptc}'
        print(msg)
        print('=' * 55)

        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount #Resets the amount if used before
        self.data['SMA'] = self.data['price'].rolling(SMA).mean()

        for bar in range(SMA, len(self.data)): #bar = row number, disregard those less than SMA window
            if self.position == 0:
                if (self.data['price'].iloc[bar] < self.data['SMA'].iloc[bar] - threshold):
                    self.go_long(bar, amount=self.initial_amount)
                    self.position = 1 #Go long if price is smaller than SMA
                elif (self.data['price'].iloc[bar] < self.data['SMA'].iloc[bar] + threshold):
                    self.go_short(bar, amount=self.initial_amount)
                    self.position = -1 #Go short if price is bigger than SMA
            elif self.position == 1:
                if self.data['price'].iloc[bar] >= self.data['SMA'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0 #Go neutral if price too high in compared to SMA
            elif self.position == -1:
                if self.data['price'].iloc[bar] <= self.data['SMA'].iloc[bar]:
                    self.place_buy_order(bar, units=-self.units)
                    self.position = 0 #Go neutral if price too low in compared to SMA
        self.close_out(bar)
```

With all things coded out, we can conveniently backtest while adjusting parameters for best performance. Keep in mind that the "best parameters" may be just a case of overfitting.


```python
bb = BacktestBase(symbol="AAPL.O", start="2010-1-1", end="2019-12-31", amount=10000, ftc = 0, ptc= 0, verbose=False)
bb.run_mr_strategy(SMA=50, threshold=5)
```

    
    
    Running mean reversion strategy | SMA=50 & thr=5
    fixed costs=0 | proportional costs=0
    =======================================================
    Final balance [$] -1295.35
    Net Performance [%] -112.95
    Trades Executed [#] 442.00
    =======================================================
    
