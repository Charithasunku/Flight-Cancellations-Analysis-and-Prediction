from sklearn.model_selection import train_test_split
from src.preprocess import load_data, get_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model, cross_validate_model

def main():
    """
    Main function to run the flight cancellation prediction pipeline.
    """
    # Load and preprocess data
    df = load_data('data/Flight.csv')
    X = df.drop('Flight_Cancelled', axis=1)
    y = df['Flight_Cancelled']
    
    preprocessor = get_preprocessor(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    best_model = train_model(X_train, y_train, preprocessor)

    # Evaluate model
    evaluate_model(best_model, X_test, y_test)
    cross_validate_model(best_model, X, y)

if __name__ == '__main__':
    main()
