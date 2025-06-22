import os
import pickle

import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

# ensures right working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_ratings(path="../data/raw/ratings.csv") -> pd.DataFrame:
    """
    Loads the ratings dataframe
    """

    df_ratings = pd.read_csv(path)

    return df_ratings


def clean_ratings_data(df_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the ratings data
    """

    n_duplicates = df_ratings.duplicated().sum()
    if n_duplicates > 0:
        print(f"Found {n_duplicates} duplicate rows, removing...")
        df_ratings = df_ratings.drop_duplicates()

    missing_values = df_ratings.isna().sum()
    if missing_values.sum() > 0:
        print(f"Found missing values: {missing_values.to_dict()}")
        df_ratings = df_ratings.dropna()

    return df_ratings


def load_books_data() -> pd.DataFrame:
    """
    Load books data
    """

    df_books = pd.read_csv("../data/raw/books.csv")

    return df_books


def clean_books_data(df_books: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the books dataframe
    """

    df_books = df_books[
        [
            "book_id",
            "authors",
            "original_publication_year",
            "original_title",
            "language_code",
            "average_rating",
        ]
    ].copy()

    eng_various = ["eng", "en-US", "en-CA", "en-GB", "en"]

    df_books["language"] = df_books["language_code"].map(
        lambda x: (
            "eng"
            if isinstance(x, str) and x in eng_various
            else x if isinstance(x, str) else "unknown"
        )
    )

    df_books.drop("language_code", axis=1, inplace=True)

    df_books.original_title = df_books.original_title.fillna(value="unkown")

    return df_books


def create_mappings(df_ratings: pd.DataFrame) -> dict[str, dict]:
    """
    Create user and item mappings
    """

    unique_users = df_ratings["user_id"].unique()
    unique_items = df_ratings["book_id"].unique()

    user_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(unique_items)}
    index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
    index_to_item = {idx: item_id for item_id, idx in item_to_index.items()}

    mappings = {
        "user_to_index": user_to_index,
        "item_to_index": item_to_index,
        "index_to_user": index_to_user,
        "index_to_item": index_to_item,
        "n_users": len(unique_users),
        "n_items": len(unique_items),
    }

    return mappings


def split_data(df_ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    split data into train and test datasets
    """

    df_ratings_train, df_ratings_test = train_test_split(
        df_ratings, test_size=0.2, random_state=11, stratify=df_ratings.user_id
    )

    return df_ratings_train, df_ratings_test


def create_interaction_matrix(
    df_ratings: pd.DataFrame, mappings: dict[str, dict]
) -> sparse.csr_matrix:
    """
    Create user-item interaction matrix
    """

    col_indices = df_ratings["book_id"].map(mappings["item_to_index"]).values
    row_indices = df_ratings["user_id"].map(mappings["user_to_index"]).values
    ratings = df_ratings["rating"].values

    user_item_matrix = sparse.csr_matrix(
        (ratings, (row_indices, col_indices)),
        shape=(mappings["n_users"], mappings["n_items"]),
    )

    return user_item_matrix


def preprocess_pipeline(output_path="../data/processed"):

    df_ratings = load_ratings()

    df_ratings = clean_ratings_data(df_ratings=df_ratings)

    mappings = create_mappings(df_ratings=df_ratings)

    with open(os.path.join(output_path, "mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)

    df_ratings_train, df_ratings_test = split_data(df_ratings=df_ratings)

    df_ratings_train.to_csv("../data/processed/df_ratings_train.csv")
    df_ratings_test.to_csv("../data/processed/df_ratings_test.csv")

    user_item_matrix_train = create_interaction_matrix(
        df_ratings=df_ratings_train, mappings=mappings
    )

    sparse.save_npz(
        os.path.join(output_path, "user_item_matrix_train.npz"), user_item_matrix_train
    )

    user_item_matrix_test = create_interaction_matrix(
        df_ratings=df_ratings_test, mappings=mappings
    )

    sparse.save_npz(
        os.path.join(output_path, "user_item_matrix_test.npz"), user_item_matrix_test
    )

    return print("done with preprocessing")


if __name__ == "__main__":
    preprocess_pipeline()
