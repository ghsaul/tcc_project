import os
import pandas as pd

def get_full_df(true_path: str, fake_path: str):
    """
    Combine the two original dataframes containing the true and fake news.
    Creates a flag column that is 1 if true and 0 if false.

    true_path: Path with the real news
    fake_path: Path with the fake news
    """
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true.drop('Unnamed: 0', axis=1, inplace=True)
    df_fake.drop('Unnamed: 0', axis=1, inplace=True)

    df_true['flag'] = 1
    df_fake['flag'] = 0

    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df[~df['text'].isnull()]

    df.to_csv(os.path.join(os.path.split(true_path)[0], 'data.csv'), index=False)

def clean_df(data: pd.DataFrame):
    """
     Remove null rows
    ,lower case all words
    ,remove signs

    data: dataframe to be cleaned
    """

    
