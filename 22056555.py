# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:18:08 2024

@author: aa23aan
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

def read_data(file_path):
    """Read data from a CSV file and return a cleaned DataFrame."""
    data = pd.read_csv(file_path, skiprows=4)
    return data


def preprocess_data(data, country_name):
    """Preprocess data, filter relevant columns, handle missing values."""
    filtered_data = data[data['Country Name'] == country_name]
    selected_data = filtered_data[['Country Name'] + \
                                  [str(year) for year in range(1976, 2021)]]
    selected_data = selected_data.dropna()
    return selected_data


def transpose_migration(data):
    """Transpose the DataFrame and reset the index."""
    transposed_data = data.transpose().reset_index()
    transposed_data['Country Name'] = data['Country Name'].values[0]
    transposed_data.columns = ['Year', 'Net Migration', 'Country Name'] + \
        list(transposed_data.iloc[0, 3:])
    transposed_data = transposed_data[1:].reset_index(drop=True)
    return transposed_data

def transpose_unemployment(data):
    """Transpose the DataFrame and reset the index."""
    transposed_data = data.transpose().reset_index()
    transposed_data['Country Name'] = data['Country Name'].values[0]
    transposed_data.columns = ['Year', 'Unemployment rate', 'Country Name'] + \
        list(transposed_data.iloc[0, 3:])
    transposed_data = transposed_data[1:].reset_index(drop=True)
    return transposed_data

def merge_transposed_data(transposed_migration, transposed_unemployment):
    """Merge transposed migration and transposed unemployment\
        data based on 'Country Name' and 'Year'."""
    merged_data = pd.merge(transposed_migration, transposed_unemployment,\
                           on=['Country Name', 'Year'])
    return merged_data

def print_silhouette_scores(data, max_clusters=10):
    """Print silhouette scores for different numbers of clusters."""
    features = data[['Net Migration', 'Unemployment rate']]
    
    silhouette_scores = []  # List to store silhouette scores
    
    for n_clusters in range(2, max_clusters + 1):
        km = KMeans(n_clusters=n_clusters)
        data['Cluster'] = km.fit_predict(features)

        # Calculate average silhouette score
        silhouette_avg = silhouette_samples(features, data['Cluster']).mean()
        silhouette_scores.append((n_clusters, silhouette_avg))

    # Print all silhouette scores together
    for n_clusters, silhouette_avg in silhouette_scores:
        print(f"For n_clusters = {n_clusters}, the average \
              silhouette score is: {silhouette_avg}")

def apply_kmeans(data, n_clusters):
    """Apply K-Means clustering and normalize relevant columns
      Also Calculates and prints the back scaled cluster center also
    ."""
    km = KMeans(n_clusters=n_clusters)
    
    # Include only 'Net Migration' and 'Unemployment Rate' for clustering
    features = data[['Net Migration', 'Unemployment rate']]
    data['Cluster'] = km.fit_predict(features)

    # Scale the relevant columns
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    
    # Print the back scaled cluster centers
    print("Unscaled Cluster Centers:")
    for i, center in enumerate(km.cluster_centers_):
        print(f"Cluster {i+1}: {center}")

    # Print the scaled cluster centers
    scaled_centers = scaler.transform(km.cluster_centers_)
    print("\nScaled Cluster Centers:")
    for i, center in enumerate(scaled_centers):
        print(f"Cluster {i+1}: {center}")

    # Update the relevant columns with scaled values
    data[['Net Migration', 'Unemployment rate']] = scaled_data

    return data



def separate_clusters(data):
    """Separate clusters based on the 'Cluster' column."""
    clusters = [data[data['Cluster'] == i] \
                for i in range(data['Cluster'].nunique())]
    return clusters

def plot_clusters(data, clusters):
    """Plot clusters with different colors and center values."""
    colors = ['royalblue', 'forestgreen', 'darkorange',\
              'red','slategrey','purple','skyblue']
    plt.figure(figsize=(11, 8))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster['Unemployment rate'], cluster['Net Migration'],\
                    color=colors[i], label=f'Cluster {i+1}')
    
    # Plot center values
    centers = data.groupby('Cluster').mean()\
        [['Unemployment rate', 'Net Migration']]
    plt.scatter(centers['Unemployment rate'], centers['Net Migration'],\
                marker='d', color='black', label='Cluster Centers')
    
    plt.xlabel('Unemployment Rate')
    plt.ylabel('Migration')
    plt.legend()
    plt.show()
    
def plot_line_graph(data, variable_name, title, x_interval=1):
    plt.figure(figsize=(10, 8))
    plt.plot(data['Year'], data[variable_name])

    plt.xlabel('Year')
    plt.ylabel(variable_name)
    plt.title(title)

    # Set x-axis interval
    plt.xticks(data['Year'][::x_interval])

    plt.show()    
    
def exponential(t, n0, g):
    """Calculates exponential function \
        with scale factor n0 and growth rate g."""
    t = t - 1990
    f = n0 * np.exp(g * t)
    return f    
    
def plot_forcast(data, variable_name, title, x_interval=3):
    # Convert the 'Year' column to numeric
    data['Year'] = pd.to_numeric(data['Year'])

    plt.figure(figsize=(10, 8))
    plt.plot(data['Year'], data[variable_name], label=variable_name)

    # Fit exponential curve
    param, covar = opt.curve_fit(exponential, data['Year'], data[variable_name])

    # Calculate uncertainty range (confidence interval) up to the end of the forecast
    year_forecast = np.linspace(data['Year'].min(), 2030, 100)
    param_std_dev = np.sqrt(np.diag(covar))
    lower_bound = exponential(year_forecast, *(param - param_std_dev))
    upper_bound = exponential(year_forecast, *(param + param_std_dev))
    
    # Plot the uncertainty range with a shaded region
    plt.fill_between(year_forecast, lower_bound,\
                     upper_bound, color='pink', alpha=0.3)

    # Plot the exponential fit with a different color and no legend
    plt.plot(data['Year'], exponential(data['Year'], *param), \
             color='red', label='_nolegend_')

    # Create array for forecasting until 2030
    year_forecast = np.linspace(data['Year'].min(), 2030, 100)
    forecast_exp = exponential(year_forecast, *param)

    # Plot the exponential forecast until 2030
    plt.plot(year_forecast, forecast_exp, label="Forecast")

    plt.xlabel("Year")
    plt.ylabel(variable_name)
    plt.title(title)
   
    # Set x-axis interval
    plt.xticks(np.arange(data['Year'].min(), 2031, x_interval))  # Ensure x-axis range includes 2030

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = 'cluster.csv'
    unemployment_file_path = 'Unemployment_National.csv'  # Replace with the actual file path
    country_name = 'Canada'
   
    data = read_data(file_path)
    selected_data = preprocess_data(data, country_name)

    # Read and preprocess unemployment data
    unemployment_data = read_data(unemployment_file_path)
    unemployment_selected_data = preprocess_data(unemployment_data, country_name)
    
    
    transposed_migration = transpose_migration(selected_data)
    transposed_unemployment = transpose_unemployment(unemployment_selected_data)
    merged_data = merge_transposed_data(transposed_migration,\
                                        transposed_unemployment)
    print_silhouette_scores(merged_data, max_clusters=10)
    
    n_clusters = 3
    clustered_data = apply_kmeans(merged_data,n_clusters)
    clusters = separate_clusters(clustered_data)
    plot_clusters(clustered_data, clusters)
    # Assuming 'merged_data' is the DataFrame containing the merged migration and unemployment data
    merged_data['Year'] = merged_data['Year'].astype(float)

    # Filter data for the years 2010-2021
    merged_data_2010_2021 = merged_data[(merged_data['Year'] >= 2010)\
                                        & (merged_data['Year'] <= 2021)]

    # Fit the curve with scaled data
    
    plot_line_graph(transposed_unemployment, 'Unemployment rate',\
                    'Unemployment over years', x_interval=3)
    plot_line_graph(transposed_migration, 'Net Migration',\
                    'Net Migration Over Years', x_interval=3)

    plot_forcast(transposed_unemployment, 'Unemployment rate', \
                 'Unemployment over years', x_interval=3)
    plot_forcast(transposed_migration, 'Net Migration', \
                 'Net Migration Over Years', x_interval=3)