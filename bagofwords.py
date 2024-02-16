import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import fitz  # PyMuPDF
import pandas as pd


stopwords_file_path = 'C:/Users/checc/OneDrive/Desktop/espy/stopwords-it.txt' 
with open(stopwords_file_path, 'r', encoding='utf-8') as file:
    custom_stopwords = [line.strip() for line in file]
nltk.download('punkt')

def pdf_to_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
    return text

def extract_words(text):
    words = word_tokenize(text)
    stop_words = set(custom_stopwords)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return filtered_words

def main():
    pdf_path = 'C:/Users/checc/OneDrive/Desktop/espy/qcn_20.pdf'

    
    text = pdf_to_text(pdf_path)
    print("Testo estratto dal PDF:")
    print(text)

    words = extract_words(text)
    bag_of_words = Counter(words)

   
    df_bag_of_words = pd.DataFrame(list(bag_of_words.items()), columns=['Parola', 'Frequenza'])
    df_bag_of_words = df_bag_of_words.sort_values(by='Frequenza', ascending=False)

    
    print("Top 10 parole più ricorrenti:")
    print(df_bag_of_words.head(10))

    csv_path = 'C:/Users/checc/OneDrive/Desktop/espy/bag_of_words.csv'
    print("Tentativo di salvare il CSV in:", csv_path)
    df_bag_of_words.to_csv(csv_path, index=False)

 

if __name__ == "__main__":
    main()
