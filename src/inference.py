import os
import pickle

import pandas as pd
from scipy import sparse

# ensures right working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


from src.preprocessing import clean_books_data, load_books_data


def load_model_and_data():
    """
    Load trained model, data and mappings
    """

    with open("../models/model1.pkl", "rb") as f:
        model = pickle.load(f)

    with open("../data/processed/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    user_item_matrix = sparse.load_npz("../data/processed/user_item_matrix_train.npz")

    return model, mappings, user_item_matrix


def recommend_for_user(
    model,
    user_item_matrix: sparse.csr_matrix,
    mappings: dict,
    user_id: int,
    n_recommendations: int = 10,
) -> list[tuple[int, float]]:
    """
    Generate recommendations for a specific user
    """

    user_to_index = mappings["user_to_index"]
    user_index = user_to_index[user_id]

    user_items = user_item_matrix[user_index]

    recommended = model.recommend(
        userid=user_index, user_items=user_items, N=n_recommendations
    )

    return recommended


def recommend_for_user_filtered(
    model,
    user_item_matrix: sparse.csr_matrix,
    mappings: dict,
    user_id: int,
    n_recommendations: int = 10,
    authors: list = None,
    publication_year_min: float = None,
    publication_year_max: float = None,
    language: list = None,
    average_rating_min: float = None,
    average_rating_max: float = None,
) -> list[tuple[int, float]]:
    """
    Generate recommendations for a specific user using filters
    """

    df_books = load_books_data()
    df_books = clean_books_data(df_books=df_books)

    item_to_index = mappings["item_to_index"]

    if authors:
        df_books = df_books[df_books.authors.isin(authors)].copy()

    if publication_year_min:
        df_books = df_books[
            df_books.original_publication_year >= publication_year_min
        ].copy()

    if publication_year_max:
        df_books = df_books[
            df_books.original_publication_year <= publication_year_max
        ].copy()

    if language:
        df_books = df_books[df_books.language.isin(language)].copy()

    if average_rating_min:
        df_books = df_books[df_books.average_rating >= average_rating_min].copy()

    if average_rating_max:
        df_books = df_books[df_books.average_rating <= average_rating_max].copy()

    unq_book_ids = df_books.book_id.unique().tolist()
    filtered_dict = {k: v for k, v in item_to_index.items() if k in unq_book_ids}
    filtered_indices = set(filtered_dict.values())
    all_indices = set(range(user_item_matrix.shape[1]))
    filter_out_items = list(all_indices - filtered_indices)

    user_to_index = mappings["user_to_index"]
    user_index = user_to_index[user_id]

    user_items = user_item_matrix[user_index]

    recommended = model.recommend(
        userid=user_index,
        user_items=user_items,
        N=n_recommendations,
        filter_items=filter_out_items,
    )

    return recommended


def format_recommendations(
    recommended: list[tuple[int, float]], mappings: dict, df_books: pd.DataFrame
) -> pd.DataFrame:
    """
    Format recommendations with more book information
    """

    index_to_item = mappings["index_to_item"]

    df_recomanded = pd.DataFrame(
        columns=[
            "score",
            "authors",
            "publication_year",
            "title",
            "language",
            "average_rating",
        ]
    )

    item_indexs = recommended[0]
    scores = recommended[1]

    for item_index, score in zip(item_indexs, scores):
        if item_index in index_to_item:
            book_id = index_to_item[item_index]
            book_row = df_books[df_books.book_id == book_id]

            df_recomanded.loc[len(df_recomanded)] = {
                "score": score,
                "authors": book_row["authors"].values[0],
                "publication_year": book_row["original_publication_year"].values[0],
                "title": book_row["original_title"].values[0],
                "language": book_row["language"].values[0],
                "average_rating": book_row["average_rating"].values[0],
            }

    return df_recomanded


def inference_pipeline(
    user_id: int = 2,
    authors: list = None,
    publication_year_min: float = None,
    publication_year_max: float = None,
    language: list = None,
    average_rating_min: float = None,
    average_rating_max: float = None,
):
    """
    Main inference pipeline
    """

    model, mappings, user_item_matrix = load_model_and_data()

    df_books = load_books_data()
    df_books = clean_books_data(df_books=df_books)

    if (
        authors
        or publication_year_min
        or publication_year_max
        or language
        or average_rating_min
        or average_rating_max
    ):
        recommended = recommend_for_user_filtered(
            model=model,
            user_item_matrix=user_item_matrix,
            mappings=mappings,
            user_id=user_id,
            authors=authors,
            publication_year_min=publication_year_min,
            publication_year_max=publication_year_max,
            language=language,
            average_rating_min=average_rating_min,
            average_rating_max=average_rating_max,
        )

    else:
        recommended = recommend_for_user(
            model=model,
            user_item_matrix=user_item_matrix,
            mappings=mappings,
            user_id=user_id,
        )

    df_recomanded = format_recommendations(
        recommended=recommended, mappings=mappings, df_books=df_books
    )

    return df_recomanded


if __name__ == "__main__":

    print(inference_pipeline(publication_year_max=1990, publication_year_min=1970))
    