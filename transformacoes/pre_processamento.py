import os
import pandas as pd

def get_processed_df(true_path: str, fake_path: str, final_path: str):
    """
    Combine the two original dataframes containing the true and fake news.
    Creates a flag column that is 1 if true and 0 if false.

    true_path: Path with the real news
    fake_path: Path with the fake news
    final_path: Path where the processed df should be stored
    """
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true.drop('Unnamed: 0', axis=1, inplace=True)
    df_fake.drop('Unnamed: 0', axis=1, inplace=True)

    df_true['flag'] = 1
    df_fake['flag'] = 0

    df = pd.concat([df_true, df_fake])
    df = df[~df['text'].isnull()]

    df['text'] = df['text'].str.lower()




    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset='text', keep=False, inplace=True, ignore_index=True)

    df.to_csv(final_path, index=False)
