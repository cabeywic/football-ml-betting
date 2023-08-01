
# Value Betting Strategies for Football
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://camo.githubusercontent.com/890acbdcb87868b382af9a4b1fac507b9659d9bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667)
[![Documentation](https://img.shields.io/badge/ref-Documentation-blue)](https://jsg71.github.io/QST_Template/)

This is a machine learning-based project aimed at optimizing football betting strategies. The project uses football data from [football-data.co.uk](https://www.football-data.co.uk/downloadm.php) and incorporates advanced statistical strategies like Elo, GAP, and Pi ratings to generate features for machine learning models. It applies different machine learning models to predict match outcomes and utilizes strategies like the Kelly Criterion and its variations to decide on the best betting approach.

The project also provides a flexible framework for testing and comparing different betting strategies, allowing for easy extension and experimentation. It includes a comprehensive historical record of bets placed, their outcomes, and the progression of the gambler's bankroll.

This project serves as a comprehensive guide for anyone interested in the intersection of machine learning, sports statistics, and strategic betting.

## Environment Variables
The conf.yml may be configured as shown below:
```yml
...
data:
  path: /Users/charaka/Desktop/University/Msc Machine Learning & Data Science/Masters Project/football-data
  division_names:
    - E0
    - E1
  years:
    - 2019-2018
    - 2020-2019
    - 2021-2022
    - 2022-2023
...
```

`path`: Path to load the football dataset from 
`division_names`: List of division names to load the data from, e.g. `E0` for the English Premier League, `E1` for the English Championship, etc.
`years`: List of years to load the data from

## Installation

Clone the project

```bash
  git clone https://github.com/cabeywic/football-ml-betting
```

After cloning the repo run the below command, to install dependencies and create a logs folder. 

```bash
cd football-ml-betting
pip install -r requirements.txt
```

## Run Locally (CLI)
Start the stream, check your configuration & environment and run below command 

```bash
  python src/main.py
```
