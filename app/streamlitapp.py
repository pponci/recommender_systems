import streamlit as st
import pandas as pd

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.inference import inference_pipeline, load_books_data
from src.preprocessing import load_ratings, clean_ratings_data, clean_books_data

def main():
    df_ratings = load_ratings()
    df_ratings = clean_ratings_data(df_ratings = df_ratings)
    df_books = load_books_data()
    df_books = clean_books_data(df_books=df_books)

    unique_users = df_ratings.user_id.unique().tolist()
    unique_authors = df_books.authors.unique().tolist()
    unique_authors = list(set([author for authors in unique_authors for author in (authors.split(",") if "," in authors else [authors])]))

    min_year_pub = abs(df_books.original_publication_year.min())
    max_year_pub = abs(df_books.original_publication_year.max())

    unique_languages = list(set(df_books.language.unique().tolist()))


    st.title("Book Recommendation System")
    st.write("Get personalized book recommendations based on your preferences!")

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Recommendation Filters")
        
        # User ID input
        user_id = st.selectbox(
            label = "Select User",
            options = unique_users
        )
        
        # Author filter
        authors = st.multiselect(
            "Filter by Authors",
            options=unique_authors,
            default=None,
            placeholder="Select authors..."
        )
        
        # Publication year range
        st.write("Publication Year Range")
        col1, col2 = st.columns(2)
        with col1:
            year_min = st.number_input("Min Year", min_value=min_year_pub, max_value=max_year_pub, value=min_year_pub, placeholder="Min", step=1.0)
        with col2:
            year_max = st.number_input("Max Year", min_value=min_year_pub, max_value=max_year_pub, value=max_year_pub, placeholder="Max", step=1.0)
        
        # Language filter
        language = st.multiselect(
            "Filter by language",
            options=unique_languages,
            default=None,
            placeholder="Select Languages..."
        )
        
        
        # Rating range
        st.write("Average Rating Range")
        col1, col2 = st.columns(2)
        with col1:
            rating_min = st.number_input("Min Rating", min_value=0.0, max_value=5.0, value=0.0, step=0.1, placeholder="Min")
        with col2:
            rating_max = st.number_input("Max Rating", min_value=0.0, max_value=5.0, value=5.0, step=0.1, placeholder="Max")

    # Convert empty inputs to None
    publication_year_min = year_min if year_min != 0 else None
    publication_year_max = year_max if year_max != 0 else None
    average_rating_min = rating_min if rating_min is not None else None
    average_rating_max = rating_max if rating_max is not None else None

    # Get recommendations when button is clicked
    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            try:
                recommendations = inference_pipeline(
                    user_id=user_id,
                    authors=authors,
                    publication_year_min=publication_year_min,
                    publication_year_max=publication_year_max,
                    language=language,
                    average_rating_min=average_rating_min,
                    average_rating_max=average_rating_max
                )
                
                st.success("Recommendations generated successfully!")
                
                # Display recommendations
                st.subheader("Recommended Books")
                st.dataframe(recommendations)
                
                # Option to download as CSV
                csv = recommendations.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download recommendations as CSV",
                    data=csv,
                    file_name='book_recommendations.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()