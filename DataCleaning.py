Q1 = df["Cholesterol"].quantile(0.25)
Q3 = df["Heart Rate"].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df["Cholesterol"] < (Q1 - 1.5 * IQR)) | (df["Heart Rate"] > (Q3 + 1.5 * IQR)))]
