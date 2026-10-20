lexploreur
=======

Distant reading library for python.

Under heavy development: for now, the package works based on annotations provided by Spacy models, but the goal is to enable the use of other NLP backends.

In particular, the data structure is likely to change in the coming weeks.



## Installation

### Prérequis

Il faut installer le gestionnaire de projet [uv](https://docs.astral.sh/uv/getting-started/installation/).

### Procédure d'installation

1. Cloner le dépôt avec `git` :

```bash
   git clone git@github.com:votre-user/lexploreur.git
   cd lexploreur
```

2. Installer les dépendances via `uv` :

```bash
   uv sync
```

3. Configure `lexploreur.py`
4. Exécuter `lexploreur.py` :

```bash
uv run lexploreur.py
```

### Alternative

If you don't want to use `uv` :

1. Install dependencies manually :

```bash
pip install pandas spacy tqdm
```
2. Configure `lexploreur.py`
3. Execute `python lexploreur.py`

### Input

A CSV file where each row is a textual document in a given column. You can have as metadata columns describing the document as you want. 

See `bulletins_république.csv` for an example.

### Output

A json file structured as follow.

See `bulletins_république.json` for a complete example.

```json
[
  { "document": { "metadata": { ... }, "content": { ... } } },
  { "document": { "metadata": { ... }, "content": { ... } } },
  ...
]
```

Document details:

```json
{
  "document": {
    "metadata": {
      "<colonne_1>": "valeur",
      "<colonne_2>": "valeur",
      "_langue_doc": "fr",
      "_nb_tokens": 42
    },
    "content": {
      "tokens": [...],
      "entities": [...],
      "sentences": [...]
    }
  }
}
```

 `content.tokens`

```json
{ "id": 0, "text": "Le", "pos": "DET", "lemma": "le", "dep": "det", "alpha": true }
```

`content.entities`

```json
{ "id": 0, "type": "PER", "texte_surface": "Rosa Bonheur", "token_start": 3, "token_end": 4 }
```

`content.sentences`

```json
{ "id": 0, "sent_start": 0, "sent_end": 11 }
```

## About the example corpus

Texts from French Provisional Government of 1848. Raw OCR from Gallica: [https://gallica.bnf.fr/ark:/12148/bpt6k56085242](https://gallica.bnf.fr/ark:/12148/bpt6k56085242)
