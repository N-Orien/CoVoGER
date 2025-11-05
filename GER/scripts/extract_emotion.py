import pandas as pd

csv_file_path = "/mnt/clover/zd-yang/resources/datasets/BMELD/BMELD_data/test_sent_emo.csv"
emotion_output_path = "/mnt/clover/zd-yang/works/EmoST_LLM/GenTranslate/results/references.emotion.zh"
sentiment_output_path = "/mnt/clover/zd-yang/works/EmoST_LLM/GenTranslate/results/references.sentiment.zh"

# Read list from the input file
#csv_data = pd.read_csv(csv_file_path, encoding='gbk')
csv_data = pd.read_csv(csv_file_path, encoding='gbk')

with open(emotion_output_path, 'w') as eop, open(sentiment_output_path, 'w') as sop:
    for _, row in csv_data.iterrows():
        emotion = row[3] 
        sentiment = row[4]
        eop.write(f"{emotion}\n")
        sop.write(f"{sentiment}\n")
