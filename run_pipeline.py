from bacpipe.main import (
    get_model_names,
    model_specific_embedding_creation,
    model_specific_evaluation,
    cross_model_evaluation,
)

if __name__ == "__main__":

    get_model_names()

    model_specific_embedding_creation()

    model_specific_evaluation()

    cross_model_evaluation()
