import pandas as pd
import spacy
import os
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import json


def model_loader(model: str, ner: bool) -> spacy.Language:
    """
    Load a Spacy language model with optional pipeline components.

    Parameters
    ----------
    model : str
        The name or path of the Spacy model to load (e.g., "fr_core_news_sm").
    ner : bool
        Flag indicating whether to enable the Named Entity Recognition component.
        If False, the "ner" pipeline is disabled.

    Returns
    -------
    spacy.Language
        The loaded Spacy nlp object.

    Raises
    ------
    OSError
        If the specified model is not installed on the system.
    """

    try:
        if ner is False:
            nlp = spacy.load(model, disable=["ner"])
        else:
            nlp = spacy.load(model)
        # nlp = spacy.load(f"{model}")
    except OSError:
        raise OSError(
            f"Modèle Spacy '{model}' non trouvé. "
            f"Veuillez l'installer avec: python -m spacy download {model}"
        )

    return nlp


def extract_tokens(doc) -> List[Dict[str, Any]]:
    """
    Extract token-level information from a Spacy document.

    Iterates through the tokens in the document and extracts the index,
    text, part-of-speech tag, and lemma for each token.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The Spacy document object to process.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents a token
        with keys: "id", "text", "pos", and "lemma".
    """
    tokens = []
    for i, token in enumerate(doc):
        tokens.append({
            "id": i,
            "text": token.text,
            "pos": token.pos_,
            "lemma": token.lemma_,
            "dep": token.dep_,
            "alpha": token.is_alpha
        })
    return tokens


def extract_entities(doc) -> List[Dict[str, Any]]:
    """
    Extract named entities with their corresponding token indices.

    Iterates through the recognized entities in the document. Note that
    the `token_end` index is adjusted by subtracting 1, as Spacy's default
    end index is exclusive, whereas this function returns the inclusive
    index of the last token in the entity.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The Spacy document object containing named entities.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents an entity
        with keys: "id", "type", "texte_surface", "token_start", and "token_end".
    """
    entities = []
    i = 0
    for ent in doc.ents:
        entities.append({
            "id": i,
            "type": ent.label_,
            "texte_surface": ent.text,
            "token_start": ent.start,
            "token_end": ent.end
        })
        i += 1
    return entities


def extract_sentences(doc) -> List[Dict[str, Any]]:
    """
    Extract sentence spans with their corresponding token indices.

    Iterates through the sentences detected by Spacy. Similar to entity
    extraction, the `sent_end` index is adjusted by subtracting 1 to
    provide the inclusive index of the last token in the sentence.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The Spacy document object to process.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents a sentence
        with keys: "id", "sent_start", and "sent_end".

    Notes
    -----
    - If the output file already exists, it will be deleted before writing.
    - Metadata columns are automatically inferred as all columns except `text_column`.
    - Progress is displayed using a tqdm progress bar.
    - Token_start/token_end and sent_start/sent_end index follow python convention
    """
    sentences = []
    i = 0
    for sent in doc.sents:
        sentences.append({
            "id": i,
            "sent_start": sent.start,
            "sent_end": sent.end
        })
        i += 1
    return sentences


def corpus(
    df: pd.DataFrame,
    corpus_name: str,
    text_column: str,
    nlp_model: str,
    ner: Optional[bool] = True,
    batch_size: int = 32,
    n_process: int = -1
) -> str:
    """
    Process a DataFrame of texts into a JSON corpus with linguistic annotations.

    This function loads a Spacy model, processes a column of text from a
    Pandas DataFrame in batches, and extracts tokens, named entities, and
    sentences. It writes the results directly to a JSON file to minimize
    RAM usage, including metadata from the original DataFrame for each document.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the text data and metadata.
    corpus_name : str
        The file path where the resulting JSON corpus will be saved.
    text_column : str
        The name of the column in `df` containing the text to process.
    nlp_model : str
        The name of the Spacy model to use for processing.
    ner : bool, optional
        Whether to enable Named Entity Recognition. Defaults to True.
    batch_size : int, optional
        The number of texts to process in each batch (default is 50).
        See https://spacy.io/api/language#pipe
    n_process : int, optional
        The number of parallel processes to use for processing (default is -1,
        which uses all available CPUs).
        See https://spacy.io/api/language#pipe

    Returns
    -------
    str
        A status message upon successful completion.

    Notes
    -----
    - If the output file already exists, it will be deleted before writing.
    - Metadata columns are automatically inferred as all columns except `text_column`.
    - Progress is displayed using a tqdm progress bar.
    """


    # D'abord on efface json si il existe déjà
    if os.path.exists(corpus_name):
        os.remove(corpus_name)

    # On vérifie que la colonne texte soumise
    # soit bien présente
    if text_column not in df.columns:
        raise OSError(
            f"Colonne '{text_column}' non trouvée. "
        )

    # On détecte d'éventuels documents sans texte
    # et on les supprime en affichant un avertissement.
    if pd.isna(df[text_column]).any():
        df.dropna(subset=[text_column], inplace=True)
        df.reset_index(inplace=True)
        print("Des documents n'ont pas de contenus textuels, ils sont supprimés.")

    # On récupère le nom des colonnes des métadonnées, càd
    # toutes sauf celle correspondant aux données textuelels
    metadata_columns = [col for col in df.columns if col != text_column]

    # On convertit chaque la colonne des données textuelles en liste
    texts = df[text_column].tolist()

    # On appelle la fonction pour charger le modèle
    nlp = model_loader(nlp_model, ner)

    # On instancie spacy pipe pour traiter des flux
    # de documents bcp plus rapidement, ici on soumet
    # la liste des données textuelles du corpus
    docs_iterator = nlp.pipe(texts,
                             batch_size=batch_size,
                             n_process=n_process)

    # Instanciation tqdm pour avoir une barre de progression
    # pendant l'exécution de l'annotation spacy
    progress_bar = tqdm(docs_iterator, total=len(texts),
                        desc="Traitement NLP", unit="doc")

    # liste des dictionnaires de chaque document
    # docs_dics = []
    # Compteur pour les id des docs
    doc_id = 0
    # On lance l'écriture du json pour écrire chaque
    # doc au fur et à mesure et éviter de tout monter
    # en RAM
    with open(corpus_name, 'w', encoding='utf-8') as f:
        # On écrit la syntaxe du json à la main
        f.write('[')
        first_doc = True  # On déclare la première itération
        # On lance l'annotation spacy via la progression tqdm
        # à chaque itération de cette boucle, on est sur un document
        # différent
        for doc in progress_bar:
            # On récupère les métadonnées pour le document dans le dataframe
            # et on les ajoute dans le dictionnaire metadata_dict
            metadata = df.loc[doc_id, metadata_columns]
            metadata_dict = metadata.to_dict()
            metadata_dict["_langue_doc"] = doc.lang_
            metadata_dict["_nb_tokens"] = len(doc)

            # On crée le dico du document avec
            # les métadonnées
            # et on récupère les infos pour chaque token
            # et celles pour les NER
            doc_json = {
                "document": {
                    "metadata": metadata_dict,
                    "content": {
                        "tokens": extract_tokens(doc),
                        "entities": extract_entities(doc),
                        "sentences": extract_sentences(doc)
                    }
                }
            }

            doc_id += 1  # incrémantation compteur doc
            # Comme on écrit le json à la main, on vérifie
            # si c'est la première itération ou non pour modifier
            # la syntaxe en fonction (virgule ou non)
            if not first_doc:
                f.write(',')
            json.dump(doc_json, f, ensure_ascii=False, separators=(',', ':'))
            first_doc = False
        f.write(']')

    return print(f"Corpus créé dans le fichier {corpus_name}.")
