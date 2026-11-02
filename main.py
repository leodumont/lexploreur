import pandas as pd
import corpus

if __name__ == "__main__":
    csvf = "/home/leo/Documents/Recherches/These/Dossiers_recherche/topos_épistémo_numérique/prosopographie/corpus_historio_prosopo.csv"

    # df = pd.read_csv(csvf)

    # corpus.corpus(df,
    #               corpus_name="corpus_historio_prosopo.json",
    #               text_column="texte",
    #               nlp_model="fr_core_news_lg",
    #               ner=True,
    #               batch_size=32,
    #               n_process=18  # attention au nombre de coeurs dispo, -1 pour tous
    #               )

    jsonf = "/home/leo/Documents/Recherches/These/Dossiers_recherche/topos_épistémo_numérique/prosopographie/corpus_historio_prosopo.json"
    vue = corpus.lexical_view(jsonf,
                              feature_to_extract="text",
                              group_by="date",
                              merge_ner=True)

    print(vue)
