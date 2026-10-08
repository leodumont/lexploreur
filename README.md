lexploreur
=======

Distant reading library for python.
Under heavy development.

## Installation

```
pip install pandas spacy tqdm
```

## Essai

Configurer `lexploreur.py`, puis `python lexploreur.py`.

## Input

A CSV file where each row is a textual document in a given column. You can have as metadata columns describing the document as you want. 

## Output

A json file structured as follow:

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
