import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def compute_dtm(
    df: pd.DataFrame,
    text_col: str = "text",
    metadata_col: str | None = None,
    min_df: int | float = 1,
    max_df: float | int = 1.0,
    sparse: bool = False,
) -> pd.DataFrame:
    """
    Compute a Document-Term Matrix from a lexical view (e.g. DataFrame
    with pre-tokenized documents).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    text_col : str
        Column name containing lists/tuples of string tokens.
    metadata_col : str, optional
        Column name to use as document identifiers (index).
    min_df : int or float
        Ignore terms appearing in fewer than min_df documents.
        If float, interpreted as proportion.
    max_df : float or int
        Ignore terms appearing in more than max_df proportion of documents.
        If int, interpreted as absolute count.
    sparse : bool
        If True, return DataFrame with sparse columns (memory-efficient).

    Returns
    -------
    pd.DataFrame
        DTM with documents as rows, terms as columns.
    """

    # On vérifie que les colonnes appelées soient bien présentes
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")
    if metadata_col and metadata_col not in df.columns:
        raise ValueError(f"Column '{metadata_col}' not found in DataFrame")

    # On vérifie que la colonne text contient bien une liste.
    # On valide par échantillon pour coût de calcul sur les grosses listes
    sample = df[text_col].head(100)
    if not all(isinstance(x, (list, tuple)) for x in sample):
        raise ValueError(f"Column '{text_col}' must contain lists/tuples of tokens")

    # Instanciation CountVectorizer de scikit-learn
    # Ici on soumet des docs déjà tokenizés d'où le analyzer=lambda x: x
    # np.int32 permet de réduire la taille mémoire, mais à voir si il faut conserver
    vectorizer = CountVectorizer(
        analyzer=lambda x: x,
        dtype=np.int32,
        min_df=min_df,
        max_df=max_df,
    )
    # On applique le vectoriseur
    sparse_matrix = vectorizer.fit_transform(df[text_col])

    # On définit l'index, soit l'id pandas, soit les modalités
    # de la variable choisie
    index = df[metadata_col].values if metadata_col else df.index

    # On récupère les labels des mots qui correspondent aux
    # colonnes
    columns = vectorizer.get_feature_names_out()

    # Selon le paramètre sparse on renvoie ou non un objet sparse_matrix
    if sparse:
        dtm = pd.DataFrame.sparse.from_spmatrix(sparse_matrix,
                                                columns=columns,
                                                index=index)
    else:
        dtm = pd.DataFrame.sparse.from_spmatrix(sparse_matrix,
                                                columns=columns,
                                                index=index)

    dtm.index.name = metadata_col

    return dtm
