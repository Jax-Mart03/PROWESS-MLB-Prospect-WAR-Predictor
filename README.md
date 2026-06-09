# PROWESS-MLB-Prospect-WAR-Predictor
Predictive baseball analytics project exploring how minor league performance translates to MLB success. Built and evaluated machine learning models using 1,000+ player seasons to predict a player's first three years of MLB WAR, with feature engineering, model comparison, and performance analysis.

# Minor League WAR Prediction

## Objective

How well can a player's prior minor league performance predict their first three years of MLB WAR?

## Data

- Lahman Database
- FanGraphs
- Baseball Reference

## Features

- Age
- Level
- OPS
- BB%
- K%
- ISO
- Stolen Bases
- Position

## Models Tested

- Linear Regression
- Random Forest
- XGBoost

## Results

Best model:

Random Forest

RMSE: 3.22
R²: 0.108

WAR is highly noisy and difficult to predict solely from minor league performance.

## Future Work

- Reduce feature redundancy 
- See impact of scouting grades
- Add Statcast metrics
- Incorporate injury history
