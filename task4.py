import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Load dataset (replace 'ratings.csv' with your dataset)
df = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv")

# Display first few rows
print(df.head())

# Define a Reader object for Surprise
reader = Reader(rating_scale=(1, 5))

# Load data into Surprise format
data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)

# Split dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD (Singular Value Decomposition) for matrix factorization
model = SVD()

# Train the model
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Evaluate RMSE
rmse(predictions)

# Function to recommend top-N books for a user
def recommend_books(user_id, model, df, n=5):
    unique_books = df['book_id'].unique()
    user_rated_books = df[df['user_id'] == user_id]['book_id'].values
    books_to_predict = np.setdiff1d(unique_books, user_rated_books)

    predictions = [model.predict(user_id, book_id) for book_id in books_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_books = [pred.iid for pred in predictions[:n]]
    return top_books

# Get recommendations for user 1
recommended_books = recommend_books(user_id=1, model=model, df=df)
print("Recommended books:", recommended_books)

