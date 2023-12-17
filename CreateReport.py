! pip install -U dataprep
import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/heart_attack_prediction_dataset.csv")
from dataprep.clean import clean_headers
clean_headers(df)
plot(df)
from dataprep.eda import plot, plot_correlation, plot_missing, plot_diff, create_report
plot_correlation(df)
from dataprep.eda import create_report
report = create_report(df, title='My Report')
report.save()
