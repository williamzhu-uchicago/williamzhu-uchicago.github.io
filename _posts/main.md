# Quantitatively Accessing Jim Cramer's Picks

When you wander around finance forums long enough, you must have heard of the Inverse Cramer strategy. Jim Cramer was an investment specialist before joining the CNBC and got his own show, *Mad Money*.

Now, say, I am a freshly hired data scientist at CNBC. I do not like Jim Cramer at all: one time I listened to his advise only to find myself losing all the money I had. I am also an ambitious person. I have read a lot of stuff regarding quantitative finance, and I believe that my huge brain that got me through Calculus can help me generate money. I want to one day take my revenge on Jim by kicking him out of his show and convincing the CEO at CNBC to let me host my own show, *Sane Money*.

What is the best way to do so as a data scientist? Well, through data, of course. In the following, I am going to:
1. Find out whether listening to his advice can actually generate me return.
2. Find out whether his stock recommendations can beat the market
3. Prove / disprove the Cramer's Law through a quasi-experiment


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
```

## Part 0: Data Clearning

I collected Jim Cramer's stock recommendation on his show from 2016 to 2022. I also collected SPY ETF as a benchmark. Before any analysis can be done, I have to clean the data first. The following actions are done:
1. Convert Jim Cramer's Calls into numerical numbers: "Positive Mention" / "Buy" as `1`, "Hold" as `0`, and "Negative Mention" / "Sell" as `-1`.
2. Drop hold recommendation given the technical difficulties in analyzing and small number of them (29) only
3. Drop S.No column
4. Transform percentage in string format to floats
5. Convert date in string format to datatime
6. Drop rows with at least 1 NA value


```python
df = pd.read_csv("cramer_picks.csv")
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S.No</th>
      <th>Company</th>
      <th>Ticker</th>
      <th>Date</th>
      <th>Call</th>
      <th>1-Day Change Recommendation</th>
      <th>1-Week Change Recommendation</th>
      <th>1-Month Change Recommendation</th>
      <th>1-Year Change Recommendation</th>
      <th>1-Day Change Benchmark</th>
      <th>1-Week Change Benchmark</th>
      <th>1-Month Change Benchmark</th>
      <th>1-Year Change Benchmark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Lululemon Athletica</td>
      <td>LULU</td>
      <td>27/3/2018</td>
      <td>Positive Mention</td>
      <td>2.0%</td>
      <td>2.7%</td>
      <td>15.4%</td>
      <td>95%</td>
      <td>-0.4%</td>
      <td>1.4%</td>
      <td>3%</td>
      <td>10%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Penn National Gaming</td>
      <td>PENN</td>
      <td>14/7/2020</td>
      <td>Buy</td>
      <td>10.3%</td>
      <td>4.4%</td>
      <td>52.5%</td>
      <td>100%</td>
      <td>-0.2%</td>
      <td>1.6%</td>
      <td>5%</td>
      <td>37%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Simon Property Group</td>
      <td>SPG</td>
      <td>13/11/2020</td>
      <td>Buy</td>
      <td>-1.2%</td>
      <td>8.2%</td>
      <td>10.6%</td>
      <td>121%</td>
      <td>0.4%</td>
      <td>-1.4%</td>
      <td>1%</td>
      <td>31%</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert Calls
df.drop(df[df.Call == "3"].index, inplace=True)
df.replace({"Positive Mention": 1, "Buy": 1, "Hold": 0, "Negative Mention": -1, "Sell": -1}, inplace=True)

# Drop hold calls
df.drop(df[df.Call == 0].index, inplace=True)

# Drop S.No
df.drop(columns="S.No", inplace=True)

# Transform percentage
for col in df.iloc[:, 4:].columns:
    df[col] = df[col].str.rstrip("%").apply(pd.to_numeric)

# Transform date
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Drop NaN
df.dropna(inplace=True)
```

## Part 1: Quick Statistical Analysis

Now let's run through some quick descriptive data. Mainly, I want to know: 
1. the number of buy/hold/sell calls over the 7 years
2. the directional accuracy of his calls on different horizons (Thats over 50% accuracy!)
3. By following his recommendations, the percentage return of our investment on different horizons (Surprisingly, you get a positive return across all horizons following his advice)
4. By following his recommendations, the percentage returns of our investment in excess of the benchmark


```python
# Number of buy, sell, and hold calls
print(df["Call"].value_counts(), "\n")

# Accuracy of his calls
day_acc = (df["Call"]*df["1-Day Change Recommendation"]>=0).sum() / len(df)
week_acc = (df["Call"]*df["1-Week Change Recommendation"]>=0).sum() / len(df)
month_acc = (df["Call"]*df["1-Month Change Recommendation"]>=0).sum() / len(df)
year_acc = (df["Call"]*df["1-Year Change Recommendation"]>=0).sum() / len(df)
print(f"1-Day Accuracy: {day_acc}\n" + \
      f"1-Week Accuracy: {week_acc}\n" + \
      f"1-Month Accuracy: {month_acc}\n" + \
      f"1-Year Accuracy: {year_acc}\n")

# Mean return
day_return = (df["Call"]*df["1-Day Change Recommendation"]).mean()
week_return = (df["Call"]*df["1-Week Change Recommendation"]).mean()
month_return = (df["Call"]*df["1-Month Change Recommendation"]).mean()
year_return = (df["Call"]*df["1-Year Change Recommendation"]).mean()
print(f"1-Day Mean Return: {day_return}\n" + \
      f"1-Week Mean Return: {week_return}\n" + \
      f"1-Month Mean Return: {month_return}\n" + \
      f"1-Year Mean Return: {year_return}\n")

# Mean return net of benchmark
day_excess_return = (df["Call"]*df["1-Day Change Recommendation"]-df["1-Day Change Benchmark"]).mean()
week_excess_return = (df["Call"]*df["1-Week Change Recommendation"]-df["1-Week Change Benchmark"]).mean()
month_excess_return = (df["Call"]*df["1-Month Change Recommendation"]-df["1-Month Change Benchmark"]).mean()
year_excess_return = (df["Call"]*df["1-Year Change Recommendation"]-df["1-Year Change Benchmark"]).mean()
print(f"1-Day Mean Excess Return: {day_excess_return}\n" + \
      f"1-Week Mean Excess Return: {week_excess_return}\n" + \
      f"1-Month Mean Excess Return: {month_excess_return}\n" + \
      f"1-Year Mean Excess Return: {year_excess_return}\n")
```

     1    14971
    -1     4923
    Name: Call, dtype: int64 
    
    1-Day Accuracy: 0.5273449281190309
    1-Week Accuracy: 0.5280486578868
    1-Month Accuracy: 0.539810998290942
    1-Year Accuracy: 0.5911330049261084
    
    1-Day Mean Return: 0.03043631245601689
    1-Week Mean Return: 0.11765356388860965
    1-Month Mean Return: 0.6506132502261989
    1-Year Mean Return: 8.560621292852117
    
    1-Day Mean Excess Return: 0.01599477229315371
    1-Week Mean Excess Return: -0.1571478837840555
    1-Month Mean Excess Return: -0.6645571529104253
    1-Year Mean Excess Return: -5.980697697798331
    
    

We already have a few interesting observations:
1. The directional accuracy of Jim Cramer is actually not bad. It also seems that the accuracy increases as our investment horizon lengthens. But this can also be due to the fact that general economic growth moves stock upwards while Jim Cramer usually only recommend buys.
2. One actually earns a positive return by listening to Jim Cramer's recommendations. However, when compared against the SPY index, his recommendation becomes pale in comparison. This seems to be in line with the general observation that no one can consistently beat the market. Actively managed funds do not fare better than the index funds.

I am particularly interested in point number 2. I want to take one step further and rigorosly find if there is a statistical significance between the mean of Jim Cramer's recommendation returns and the benchmark returns. Well, one simple way to achieve this is through comparing between two means and computing a t-statistic.

Consider the 1-day change case. Let $X_1, ..., X_n$ be the random variable recording the 1-day price change following Jim Cramer's recommendations, and let $Y_1, ..., Y_m$ be the random variable recording the 1-day price change buying directly the SPY index fund. Here, I assume the random variables are independent of each other. Note that the independence assumption is for the 1-day change case or even the 1-week change case is valid but may not be true if we expand our horizon to 1 month or 1 year (since they present a lot of overlaps for variables $Y_i$). We need not to know the distribution of these random variables, since we can apply the law of large numbers and t-statistics regardless of whether they are normal or not.

Suppose the true mean return of $X_i's$ is $\mu$ and the true mean return of $Y_i's$ is $\theta$. Then my hypothesis is the following:
$$
\left\{
\begin{array}{ll}
      H_0: \mu - \theta > 0 \\
      H_1: \mu - \theta \leq 0
\end{array} 
\right.
$$ 

With the help of my STAT 24510 Midterm, I can derive a rejection region:
$$
\bar{X} - \bar{Y} < \frac{t_{n+m-2, 1-\alpha}}{\sqrt{\frac{nm}{n+m}}}\sqrt{\frac{1}{n+m-2}[\sum_{i=1}^n(X_i-\bar{X})^2+\sum_{j=1}^m(Y_j-\bar{Y})^2]}
$$

Reference: http://www.stat.yale.edu/Courses/1997-98/101/meancomp.htm


```python
ALPHA = 0.01

def t_test(df, alpha, horizon):
    x = (df["Call"]*df[f"1-{horizon} Change Recommendation"]).dropna()
    y = (df[f"1-{horizon} Change Benchmark"]).dropna()
    n, m = len(x), len(y)
    sum_of_square = ((x-x.mean())**2).sum() + ((y-y.mean())**2).sum()
    t_stat = stats.t.ppf(1-alpha, n+m-2) / np.sqrt(n*m/(n+m)) * np.sqrt(1/(n+m-2) * sum_of_square)
    if x.mean()-y.mean() < t_stat:
        print("REJECT Null Hypothesis")
    else:
        print("NO REJECTION")

t_test(df, ALPHA, "Day") #horizon: "Day" / "Week"
```

    REJECT Null Hypothesis
    

From the above, we can already see that following Jim Cramer's advices will not give you a statistically significant market-beating performance on both day and week horizons. The same test is difficult to conduct on months and year horizons because durations to calculate price changes overlap, leading to dependency between variables. Nevertheless, mean excess returns of less than 0 already indicate that it is not better than the market.

OK, so what about the infamous "Inverse Cramer" Strategy? Will it give us a statistically significant market-beating performace? Well, by multiplying the "Call" column with `-1` and repeating the above, we have our result: a big fat no.


```python
ALPHA = 0.05

df_inverse = df.copy(deep=True)
df_inverse["Call"] = df_inverse["Call"]*(-1)
t_test(df_inverse, ALPHA, "Week")
```

    REJECT Null Hypothesis
    

## Part 2: A Difference-in-Difference Experiment

Well, looks like I have one thing to attack Jim Cramer on: his recommendation do not beat the market. But this point is pretty weak: many money managers consistently fail to beat the market but they are still sitting on piles of clients' money. Also, let's be honest: nobody really watches his show for serious stock recommendation, we have Elon Musk's Twitter for that. We watch it because Jim Cramer is a funny character. But I have another thing to attack him on: the Cramer's Law. The law states that whenever Jim Cramer recommends a stock, it will quickly fail to its knee because of the recommendation. Well, although this law is more like a meme than a law, but a man can hope.

Here, I adopted a simple difference-in-differences (DiD) approach. The DiD method is a quasi-experimental approach that compares the changes in outcomes over time between the treatment group and the comparison group. One way to conduct a DiD experiment is via regression, which also has the advantages of adding control variables to rule out confounders if needed. The equation for regression is:

$$
y = \beta_{0} + \beta_{1}x_{group} + \beta_{2}x_{post} + \beta_{3}(x_{group} \cdot x_{post}) + \epsilon
$$

where $x_{group}$ and $x_{post}$ are indicator variables, equating to 1 if the data point is in the treatment group and if the data point is observed post-treatment respectively. Effect of Jim Cramer's recommendation can thus be isolated and indicated only in $\beta_{3}$.

Reference: https://timeseriesreasoning.com/contents/introduction-to-the-difference-in-differences-regression-model/

[TODO] *The actual experiment is yet to be done. One primary difficulty is identifying pairs of stock such that the parallel assumption can hold.*
