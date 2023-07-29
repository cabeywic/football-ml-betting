from piny import YamlLoader
import logging
import dotenv
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from utils import setup_logging
from data.loader.football_data import FootballData
from data.feature import EloRatingGenerator, WinStreakGenerator, GapRatingGenerator, PiRatingGenerator
from models.classifier_comparison import ClassifierComparison
from strategy.kelly_criterion import KellyCriterionStrategy


def main():
    logging.info("Loading config file...")
    config = YamlLoader(path="./config/config.yaml").load()
    FOOTBALL_DATA_CONFIG = config["data"]
    COLUMNS = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA']

    logging.info("Loading football data...")
    football_dataset = FootballData.load(**FOOTBALL_DATA_CONFIG, columns=COLUMNS)
    elo_rating_generator = EloRatingGenerator()
    win_streak_generator = WinStreakGenerator()
    gap_rating_generator = GapRatingGenerator(input_features=[("FTHG", "FTAG"), ("HC", "AC")])
    pi_rating_generator = PiRatingGenerator()

    football_dataset.compute_features([elo_rating_generator, win_streak_generator, gap_rating_generator, pi_rating_generator])

    label_encoder = LabelEncoder()
    feature_columns = ['HomeEloBefore', 'AwayEloBefore', 'B365H', 'B365A', 'B365D', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'HomeGapAttackBefore', 
                    'HomeGapDefendBefore', 'AwayGapAttackBefore', 'AwayGapDefendBefore','HomeStreakBefore', 'AwayStreakBefore', 'HomePiBefore', 'AwayPiBefore']

    division = football_dataset.divisions['E0']

    logging.info("Preparing dataset")
    df = division.dataframe
    df = df.dropna(subset=feature_columns)
    df['FTR'] = label_encoder.fit_transform(df['FTR'])

    model_features = ['HomePiBefore', 'AwayPiBefore', 'B365H', 'B365A', 'B365D', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA']
    X = df[model_features]
    y = df['FTR']

    # Split the data into train and test sets
    logging.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        # 'SVC': make_pipeline(StandardScaler(), SVC()),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    classifier_comparison = ClassifierComparison(classifiers)
    classifier_comparison.fit_and_score(X_train, y_train, X_test, y_test)
    classifier_comparison.plot_scores()
    best_classifier = classifier_comparison.best_classifier()
    logging.info(f"Scores: {classifier_comparison.scores}")
    logging.info(f"Best classifier: {best_classifier} | Score: {classifier_comparison.scores[best_classifier]}")

    strategy = KellyCriterionStrategy(label_encoder)
    logging.info(f"Start Bankroll[{strategy.name}]: {strategy.bankroll}")
    logging.info(f"Running strategy[{strategy.name}]...")
    strategy.run(X_test, y_test, classifiers[best_classifier])
    logging.info(f"Final Bankroll[{strategy.name}]: {strategy.bankroll}")


if __name__ == "__main__":
    # Adds higher directory to python modules path.
    sys.path.append(".")
    # Load environment variables from .env file
    dotenv.load_dotenv()
    # Load & setup logging config
    setup_logging('./config/logging.yaml', log_dir="../logs/")

    main()
