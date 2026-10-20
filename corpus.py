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


def lexical_view(
    corpus_file: str,
    feature_to_extract: str = 'text',
    lowercase: bool = False,
    stopwords: Optional[List[str]] = None,
    exclude_tokens: Optional[List[str]] = None,
    exclude_pos: Optional[List[str]] = None,
    exclude_lemmas: Optional[List[str]] = None,
    exclude_ner_types: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    merge_ner: bool = False,
) -> pd.DataFrame:
    """
    Extract lexical units from an annotated corpus (spaCy JSON format)
    and aggregate them into a DataFrame.

    Each document in the corpus is reduced to the list of lexical units
    selected according to the requested feature, after applying the various
    filters (stopwords, tokens, POS, lemmas, named entity types). Units can
    optionally be merged around named entities, and documents can be grouped
    through a metadata field.

    Parameters
    ----------
    corpus_file : str
        Path to the corpus JSON file. Each document must follow the
        ``document.content.tokens`` schema (with ``id``, ``text``, ``pos``,
        ``lemma``) and, optionally, ``document.content.entities`` (with
        ``type``, ``texte_surface``, ``token_start``, ``token_end``).
    feature_to_extract : {'text', 'pos', 'lemma'}, default 'text'
        Lexical feature to extract for each token.
    lowercase : bool, default False
        If True, the extracted units are converted to lowercase. Does not
        affect filter comparison, which is always case-insensitive.
    stopwords : list of str, optional
        List of words to exclude (compared in lowercase).
    exclude_tokens : list of str, optional
        List of tokens to exclude (compared in lowercase).
    exclude_pos : list of str, optional
        List of POS labels (``pos``) to exclude.
    exclude_lemmas : list of str, optional
        List of lemmas to exclude (compared in lowercase).
    exclude_ner_types : list of str, optional
        Named entity types to exclude (e.g. ``['PER', 'LOC']``).
        Comparison is case-insensitive. The exclusion removes all tokens
        covered by the entity, whether ``merge_ner`` is enabled or not.
    group_by : str, optional
        Name of a metadata field (under ``document.metadata``) used as a
        grouping key. If provided, the unit lists of documents sharing the
        same value are concatenated.
    merge_ner : bool, default False
        If True, the tokens of a same named entity are merged into a single
        unit. The value of the merged unit depends on ``feature_to_extract``:
        ``'ENT'`` for ``'pos'``, otherwise the entity's ``texte_surface``.

    Returns
    -------
    pandas.DataFrame
        DataFrame of the extracted lexical units. A column named after
        ``feature_to_extract`` holds the list of units. If ``group_by`` is
        provided, an additional column bears that name and the result is
        aggregated per group.

    Raises
    ------
    FileNotFoundError
        If ``corpus_file`` does not exist.
    ValueError
        If ``feature_to_extract`` is not one of
        ``{'text', 'pos', 'lemma'}``, or if the corpus
        file does not contain valid JSON.

    Notes
    -----
    - The ``stopwords``, ``exclude_tokens`` and ``exclude_lemmas`` filters
      are compared in lowercase; ``exclude_ner_types`` is compared in
      uppercase (spaCy convention: PER, LOC, ORG, MISC...).
    - When an entity is merged, the ``stopwords``, ``exclude_tokens`` and
      ``exclude_lemmas`` tests apply on the merged value, while
      ``exclude_pos`` still relies on the ``pos`` of the entity's first token.
    - ``exclude_ner_types`` works independently of ``merge_ner``: the internal
      tokens of excluded entities are removed in both modes.

    Examples
    --------
    >>> df = lexical_view('test.json', feature_to_extract='text',
                          lowercase=True, group_by='auteur')
    >>> df = lexical_view('test.json', feature_to_extract='text',
                          merge_ner=True, exclude_ner_types=['LOC'])
    """

    # On vérifie que le fichier json du corpus existe
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    # On vérifie que l'unité lexicale demandée soit dans la
    # liste des possibles
    if feature_to_extract not in ('text', 'pos', 'lemma'):
        raise ValueError(f"feature_to_extract must be 'text', 'pos' or 'lemma', given : {feature_to_extract}")

    # On gère un éventuel json non valide
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in corpus file: {e}")

    # Normalisation des listes de comparaison
    # Ici on crée des ensembles pour réduire le nombre de valeurs
    # à vérifier. Si pas d'unités à filtrer, on fait des ensembles vides
    stopwords = set(w.lower() for w in stopwords) if stopwords else set()
    exclude_tokens = set(t.lower() for t in exclude_tokens) if exclude_tokens else set()
    exclude_pos = set(exclude_pos) if exclude_pos else set()
    exclude_lemmas = set(l.lower() for l in exclude_lemmas) if exclude_lemmas else set()
    exclude_ner_types = set(t.upper() for t in exclude_ner_types) if exclude_ner_types else set()

    # On définit une liste qui va contenir les lignes
    # du dataframe final
    rows = []

    # On parcourt chaque doc
    for doc in data:
        # On se déplace dans le dicon du json juqu'au noeud des tokens
        tokens = doc['document']['content']['tokens']
        # Idem pour les NER (entities ici)
        entities = doc['document']['content'].get('entities', [])

        # On crée un ensemble pour stocker les id des tokens à
        # exclure sous la forme de tuple
        # (index premier token, index dernier token).
        excluded_token_ids = set()

        # Dictionnaire pour gérer la fusion des tokens
        # en cas d'entités nommées composées.
        # La clé est l'index du premier tokens de l'entité
        # et la valeur est un tuple de type
        # (text_surface, index du dernier token)
        # Par exemple ner_map = {1: ('Paris', 1)}
        ner_map = {}

        # On parcourt chaque entité
        for ent in entities:
            # Si le type de l'entité fait partie de ceux à exclure
            # on l'ajoute dans le set excluded_token_ids
            if ent['type'] in exclude_ner_types:
                excluded_token_ids.update(range(ent['token_start'],
                                                ent['token_end'] + 1))
            elif merge_ner:
                ner_map[ent['token_start']] = (ent['texte_surface'],
                                               ent['token_end'])

        # On crée une liste pour stocker les formes filtrées qui
        # seront déterminées à la fin de chaque itération sur les tokens
        # un peu plus bas.
        filtered = []

        # variable pour enregistrer la position de fin du
        # token parcouru en cas d'entité à fusionner.
        # On fixe la valeur à -1 pour que les id à 0
        # ne soient pas sautés à tort
        skip_until = -1

        # On parcourt chaque tokens
        for t in tokens:
            # on récupère l'id du token parcouru
            tid = t['id']

            # Tant que l'id courant ne dépasse pas la
            # borne skip_until, on saute le token car
            # il appartient à une entité déjà rendue.
            # Rappel : continue en python signifie qu'on
            # met fin à l'itération en cours
            if tid <= skip_until:
                continue

            # On passe si l'id correspond à un token
            # à exclure.
            if tid in excluded_token_ids:
                continue

            # Si l'id est dans le dico des entités à
            # fusionner on récupère sa surface et son
            # index de fin pour les assigner
            # aux variables surface et end
            if tid in ner_map:
                surface, end = ner_map[tid]
                # on change la valeur de la borne
                skip_until = end
                # si on renvoie les pos, on affiche
                # ENT pour l'entité correspondant
                if feature_to_extract == 'pos':
                    val = 'ENT'
                # Sinon (donc pour lemma et text) on
                # renvoie les tokens qui la composent
                else:
                    val = surface
            # Si l'id n'est pas dans la dico des entités
            # à fusionner, on récupère la valeur du token
            else:
                val = t[feature_to_extract]

            # Pour gérer les filtres, on stock dans une
            # variable provisoire le token parcouru en miniscule
            val_cmp = val.lower()

            # Si le token parcouru fait partie d'une des listes
            # d'exlclusion, on interrompt l'itération pour passer
            # à la suivante.
            if val_cmp in stopwords:
                continue
            if val_cmp in exclude_tokens:
                continue
            if t['pos'] in exclude_pos:
                continue
            if t['lemma'].lower() in exclude_lemmas:
                continue

            # Si le token parcouru ne doit pas être filtré,
            # on l'ajoute à la liste filtered (en miniscule ou non
            # selon l'option choisie)
            filtered.append(val.lower() if lowercase else val)

        # On crée un dico row avec les tokens filtrés pour le
        # document parcouru
        row = {feature_to_extract: filtered}

        # Si option group_by est activée, on ajoute la métadonnée
        # concernée dans le dictionnaire row
        if group_by:
            row[group_by] = doc['document']['metadata'][group_by]

        # On ajoute le dictionnaire row du document à la liste
        # rows créée en amont de la boucle sur chaque doc
        rows.append(row)

    # à partir de cette liste de dictionnaires on crée un
    # dataframe pandas
    df = pd.DataFrame(rows)

    # Si option group_by, on utilise sum pour concaténer les documents
    # qui partagent la même valeur de métadonnées
    if group_by:
        df = df.groupby(group_by)[feature_to_extract].sum().reset_index()

    return df
