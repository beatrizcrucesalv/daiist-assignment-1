# Assignment 1 Report

- **Name**: Beatriz Cruces Alvaro
- **Student ID**: 20082
- **Email**: bcruces.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 

## Dataset
 It is a bike sharing data set. I chose it because this data is based on real business problem, predicting how many bikes can be rented at different times in a day. I thought it could create a solution for a company's problem in real life. It is also a good dataset for feature engeneering because demand changes throughout the day and it isnt a simple pattern. 

Where I got it from: I found it on kaggle "Bike Sharing Dataset". But the dataset originally is from the UCI Machine Learning Repository. It has hour rental data from Washington DC. 

Each row is one hour of one day. The target is `cnt`, the total number of bikes rented in that hour (casual + registered).

Size: 17,379 rows and 17 columns, covering from 2011 to 2012. 
I used the hourly file (hour.csv) and not the daily one (day.csv), because it had more important data for training.


Citation: Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 1-15. doi:10.1007/s13748-013-0040-3

## Business / real-life framing


This could serve several scenarios. One that I think would give value to a company running a bicycle service is in transportation, that is, the distribution of bicycles, both in timing and in quantity. A company can estimate demand for the following day so it can plan its capacity. If demand is underestimated, customers can't find bikes and the rental is lost.

I split the data 60% for training, 20% for validation and 20% for test, in chronological order, so that the model learns from the past and predicts future observations. I think this is the better way to explain patterns and validate them, since a random split would be unrealistic. 


The target is `cnt`, which is continuous, that's why I used regression and not classification. I kept the target on its normal scale because the peaks at certain hours are very important to spot. Negative predictions are set to zero, because there can't be negative rentals.

And the three models will use the same metric. 

The main evaluation is done with RMSE, which penalises large prediction errors. In this case, under-predicting costs more than over-predicting: missing very busy hours can cost more than having spare supply in non-busy hours. I also use MAE as a secondary metric, to understand the average prediction error. To see in which direction the model tends to fail, either over-predicting or under-predicting, I also look at the signed error.


## Data preparation & feature engineering

Before the training, I checked the data fro missing values, or duplicates. But it didnt have any. But some random hours are missing in these two years. I havent created more observations, I kept the ones i had because my model doent use previous-hour rentals as input, so those do not affect training directly. 

After this i removed the variables casual and registered from inputs because i felt these didnt give relevant information. Same with instant variable. 

There are some observations where humidity was 0, all from the same day. I thought this was an error, because there was no real day with 0 humidity at any hour, so I replaced them with the median humidity from the training set. Three observations were from weather category 4 (heavy rain or snow with fog); if I kept them, the model wouldn't be able to train and test correctly because there is not enough data, so I merged them into weather category 3. There were also 0 values in wind speed, but I kept those, because they could represent calm conditions. And I left out atemp, because it was very correlated with temp, so I thought it was redundant.

The most important feature for me was representing the hour of the day correctly. So I used one-hot encoding for this, instead of treating it as a number, because otherwise the model would assume that demand increases equally between hours. With one-hot encoding each hour has its own indicator, so the model can learn a different demand for each hour, rather than assuming that demand changes at a constant rate. So it can learn the busier hours. I also included some interactions between the hour, working-day status, temperature and year. I felt demand changed a lot with the weather and temperature.

The final feature has 156 numerical features. And I standarised them using means and stdev. This is to help gradient descent work beter in pyTorch. And I did it in the three models. 

I changed the code several times, to see which variables were the most important and I realised that adding hour interactions with temperature and year, separated by working status reduced validation RMSE from 113.9 to 80.5 bikes per hour. Instead adding atemp or other variable didnt help much.


## Modeling: three implementations, one model

I am using a linear regression because the target variable is the number of bikes rented per hour, which is continuous, not binary.

I implemented a basic linear model in three ways: using scikit-learn, writing the loop manually in PyTorch, and using the torch.nn.Module/torch.optim workflow. The scikit-learn implementation uses Ridge regression, which is linear regression with a penalty on large weights. In the manual PyTorch implementation, I defined the weights and bias directly and updated them using gradient descent. In the standard PyTorch implementation, I used nn.Linear for the model and torch.optim.SGD to update its parameters. All three use the same target, features, data split and evaluation metrics.

I compared the models against two simple baselines, predicting the average number of rentals in the training set for every hour, and predicting the training set average for each hour of the day.

The results showed that the three implementations work very similarly, and outperform the baselines. The final model has a test RMSE of 93.3 bikes per hour, compared with 243.7 for the training mean baseline. This suggests that the model learns useful patterns in hourly demand rather than simply predicting an average value.


All three implementations give the same results because they fit the same linear model and minimise the same mean squared error. The manual PyTorch loop and the standard PyTorch use the same gradient descent update, the only difference is that the standard version updates parameters by itself. In the final version, these differences are very small, teh differences are of 0.01 bikes between implementations.

However at the begining this wasnt the case, with larger feature set, PyTorch gave different results because it needed more training. After increasing training to 10,000 epochs with a learning rate of 0.2, their results became almost identical to scikit-learn, while a learning rate of 0.3 was too high and caused training to fail.

The model still underestimates demand during commuting hours, so teh problem isnt fully fixed with this model, on average it predicts about 71.5 fewer bikes than are actually rented. It is better than the 129.6 bikes that weren't predicted before. 

## Limitations & next steps

The higher limitation is the underprediction it does of 71 bikes, so I would give more importance to peak-hour predictions. Also, the model uses actual weather data, which may not be available when making future predictions. I would address this by using weather forecasts instead. The model also uses data from Washington DC in 2011–2012, so I would test it with more recent data to check whether its predictions remain accurate.

The dataset only gives total rentals for the whole system, not demand at individual stations. So the model can estimate how many bikes will be rented in a given hour, but not where they will be needed, which means it can’t tell us how to redistribute bikes. With station-level data, I would predict demand for each location separately, which would make the model far more useful for operational planning.


## Generative AI use disclosure

I used generative AI to help organise the report, improve the wording, understand and help me make some decisions choosing the right features. 

