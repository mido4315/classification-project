Q1 = df["failures"].quantile(0.25)
Q3 = df["failures"].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df["failures"] < (Q1 - 1.5 * IQR)) | (df["failures"] > (Q3 + 1.5 * IQR)))]
