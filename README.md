lexploreur
=======

Distant reading library for python.

Under heavy development: for now, the package works based on annotations provided by Spacy models, but the goal is to enable the use of other NLP backends.

In particular, the data structure is likely to change in the coming weeks.



## Installation

```
pip install pandas spacy tqdm
```

## Usage

Configure `lexploreur.py`, then `python lexploreur.py`.

### Input

A CSV file where each row is a textual document in a given column. You can have as metadata columns describing the document as you want. 

See `corpus.csv` for an example.

### Output

A json file structured as follow.

See `corpus.json` for a complete example.

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
