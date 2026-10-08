import pandas as pd
import corpus

if __name__ == "__main__":
    csvf = "corpus.csv"

    df = pd.read_csv(csvf)

    corpus.corpus(df,
                  corpus_name="corpus.json",
                  text_column="textss",
                  nlp_model="fr_core_news_lg",
                  ner=True,
                  batch_size=32,
                  n_process=4  # attention au nombre de coeurs dispo, -1 pour tous
                  )
