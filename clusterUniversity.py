# Import panda and sklearn
import pandas as pd
from sklearn.cluster import KMeans

# Load the input data from a CSV file into data frames
data = pd.read_csv(r"C:\Users\USER\PycharmProjects\pythonProject4\College.csv")

# Convert the 'Private' column from binary categorical to numerical form
data['Private'] = data['Private'].apply(lambda x: 1 if x == 'Yes' else 0)

# Create a KMeans clustering model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Fit the model to the 'Private' columnx
kmeans.fit(data[['Private']])

# Predict the cluster labels for each row of the input data
cluster_labels = kmeans.predict(data[['Private']])

# Create a new DataFrame with the university names and corresponding cluster labels
result = pd.DataFrame({'University': data['University names'], 'Group': cluster_labels})

# Replace the cluster labels with 'Private' or 'Public' depending on the numerical value
result['Group'] = result['Group'].apply(lambda x: 'Private' if x == 1 else 'Public')

# Save the result DataFrame to a csv file
result.to_csv('output.csv', index=False)

# Print a message to show the output has been saved
print('Cluster labels saved to output.csv')
