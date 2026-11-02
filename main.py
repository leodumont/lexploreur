import pandas as pd
from lexploreur.corpus import corpus
# from lexploreur.corpus import lexical_view

if __name__ == "__main__":
    csvf = "bulletins_république.csv"

    df = pd.read_csv(csvf)

    corpus(df,
           corpus_name="bulletins_république.json",
           text_column="textes",
           nlp_model="fr_core_news_lg",
           ner=True,
           batch_size=32,
           n_process=10  # attention au nombre de coeurs dispo, -1 pour tous
           )

    # jsonf = "bulletins_république.json"
    # vue = corpus.lexical_view(jsonf,
    #                           feature_to_extract="text",
    #                           group_by="date",
    #                           merge_ner=True)

    # print(vue)
