import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import io

df = pd.read_csv("./records/latency_records_whisper_model.csv")

df['latency'] = df['latency'].str.replace('초', '').astype(float)
df = df[df['task'] != 'e2e'] # drop e2e column

stats = df.groupby('task')['latency'].agg(['count', 'mean', 'std', 'min', 'max'])

stats.to_csv('./records/task_latency_stats_whisper_model.csv')

#plt.figure(figsize=(14, 10))
#sns.boxplot(x='task', y='latency', data=df)
#plt.title('Latency Distribution')
#plt.show()