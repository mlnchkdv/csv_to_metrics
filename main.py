import streamlit as st
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt')

# Initialize ROUGE
rouge = Rouge()

def calculate_metrics(row, reference_col, compare_col):
    reference = row[reference_col]
    compare = row[compare_col]
    
    # BLEU
    bleu = sentence_bleu([reference.split()], compare.split())
    
    # ROUGE
    rouge_scores = rouge.get_scores(compare, reference)[0]
    rouge_1 = rouge_scores['rouge-1']['f']
    rouge_2 = rouge_scores['rouge-2']['f']
    rouge_l = rouge_scores['rouge-l']['f']
    
    return bleu, rouge_1, rouge_2, rouge_l

def main():
    st.title("NLP Metrics and Visualization App")
    
    st.write("""
    ## How to use this app:
    1. Upload a CSV file containing columns of text (author's abstract and translations).
    2. Rename columns if necessary and select the reference column.
    3. Calculate metrics (BLEU, ROUGE) for each translation.
    4. View visualizations and download the updated CSV with metrics.
    
    This app helps you compare different translations and analyze their quality using various NLP metrics.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Original DataFrame:")
        st.write(df)
        
        # Column renaming
        st.subheader("Rename Columns")
        new_column_names = {}
        for col in df.columns:
            new_name = st.text_input(f"Rename column '{col}'", col)
            new_column_names[col] = new_name
        
        df = df.rename(columns=new_column_names)
        
        # Select reference column
        reference_col = st.selectbox("Select reference column", df.columns)
        
        # Calculate metrics
        if st.button("Calculate Metrics"):
            for col in df.columns:
                if col != reference_col:
                    df[f'{col}_BLEU'], df[f'{col}_ROUGE-1'], df[f'{col}_ROUGE-2'], df[f'{col}_ROUGE-L'] = zip(*df.apply(lambda row: calculate_metrics(row, reference_col, col), axis=1))
            
            st.write("DataFrame with Metrics:")
            st.write(df)
            
            # Download updated CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV with metrics",
                data=csv,
                file_name="nlp_metrics.csv",
                mime="text/csv",
            )
        
        # Visualizations
        st.subheader("Visualizations")
        
        # Word cloud
        st.write("Word Cloud of Reference Text")
        text = ' '.join(df[reference_col])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        # Frequency analysis
        st.write("Top 10 Most Frequent Words")
        words = nltk.word_tokenize(text.lower())
        freq_dist = nltk.FreqDist(words)
        top_words = freq_dist.most_common(10)
        
        fig, ax = plt.subplots()
        sns.barplot(x=[word for word, freq in top_words], y=[freq for word, freq in top_words])
        plt.xticks(rotation=45)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        st.pyplot(fig)
        
        # Metrics comparison
        st.write("Metrics Comparison")
        metrics_df = df[[col for col in df.columns if any(metric in col for metric in ['BLEU', 'ROUGE'])]]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=metrics_df)
        plt.xticks(rotation=90)
        plt.ylabel('Score')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
