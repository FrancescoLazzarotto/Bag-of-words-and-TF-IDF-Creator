import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

def calculate_tfidf(input_csv, output_csv):

    df_bag_of_words = pd.read_csv(input_csv)

    label_encoder = LabelEncoder()
    df_bag_of_words['Parola'] = label_encoder.fit_transform(df_bag_of_words['Parola'])


    data = df_bag_of_words[['Parola', 'Frequenza']]


    tfidf_transformer = TfidfTransformer(use_idf=False)
    tfidf_matrix = tfidf_transformer.fit_transform(data)


    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_transformer.get_feature_names_out())


    df_tfidf[['Parola', 'Frequenza']] = df_bag_of_words[['Parola', 'Frequenza']]


    df_tfidf.to_csv(output_csv, index=False)
    print(f"Il file con i dati TF-IDF è stato salvato in: {output_csv}")

if __name__ == "__main__":
    input_csv_path = 'C:/Users/checc/OneDrive/Desktop/espy/bag_of_words.csv'
    output_csv_path = 'C:/Users/checc/OneDrive/Desktop/espy/tfidf_data.csv'

    calculate_tfidf(input_csv_path, output_csv_path)
