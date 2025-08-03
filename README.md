

# Flight Cancellation Prediction

This project uses a Random Forest Classifier to predict the likelihood of a flight being canceled based on historical flight data. The model is optimized using `GridSearchCV` to achieve high accuracy.

## Project Structure

The repository is organized as follows:



```
├── data/
│   └── Flight.csv
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── main.py
└── requirements.txt
```




-   **`data/`**: Contains the dataset.
-   **`src/`**: Contains the source code for preprocessing, training, and evaluation.
-   **`main.py`**: The main script to run the entire pipeline.
-   **`requirements.txt`**: Lists the required Python packages.

## How to Run

1.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the main script:
    ```bash
    python main.py
    ```

## Model & Performance

A Random Forest Classifier was trained and tuned to predict flight cancellations. The final model achieved the following performance on the test set:

### **Results**

-   **Accuracy**: `0.9867`

-   **Classification Report**:
    ```
                  precision    recall  f1-score   support

               0       0.96      1.00      0.98       187
               1       1.00      0.98      0.99       413

        accuracy                           0.99       600
       macro avg       0.98      0.99      0.98       600
    weighted avg       0.99      0.99      0.99       600
    ```

-   **Confusion Matrix**:
    ```
    [[187   0]
     [  8 405]]
    ```
````
