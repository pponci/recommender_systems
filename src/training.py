import os
import pickle

from implicit.als import AlternatingLeastSquares
from scipy import sparse

# ensures right working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_training_data():
    """
    Load preprocessed training data
    """

    user_item_matrix = sparse.load_npz("../data/processed/user_item_matrix_train.npz")

    return user_item_matrix


def create_model(
    factors: int = 100,
    regularization: float = 0.01,
    iterations: int = 50,
    random_state: int = 11,
):
    """
    Create and configure the recommendation model
    """

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )

    return model


def train_model(model, train_matrix: sparse.csr_matrix):
    """
    Train the recommendation model
    """

    model.fit(train_matrix)

    return model


def save_model(model, model_save_path="../models/model1.pkl"):
    """
    Save the trained model
    """

    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)


def training_pipeline():
    """
    Main training pipeline
    """

    user_item_matrix = load_training_data()

    model = create_model()

    model = train_model(model=model, train_matrix=user_item_matrix)

    save_model(model=model)

    print("model trained and saved")


if __name__ == "__main__":
    training_pipeline()
