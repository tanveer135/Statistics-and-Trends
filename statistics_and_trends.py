"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""

from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Creates a scatter plot showing Sales vs Order Date
    and a line plot showing average Sales per month.
    Saves the plots as 'relational_plot.png'.
    """

    # Convert Order Date to datetime format
    df['Order Date'] = pd.to_datetime(
        df['Order Date'], dayfirst=True, errors='coerce'
    )
    
    # Sort data by Order Date
    df_sorted = df.sort_values('Order Date')
    
    # Create a Month column for grouping
    df['Month'] = df['Order Date'].dt.to_period('M')
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Scatter plot: Sales vs Order Date
    daily_sales = df_sorted.groupby('Order Date')['Sales'].sum()

    sns.lineplot(
        ax=axes[0], x=daily_sales.index, y=daily_sales.values
    )
    axes[0].set_title('Daily Sales Over Time')
    axes[0].set_xlabel('Order Date')
    axes[0].set_ylabel('Total Sales')
    
    
    # Line plot: Average Sales Per Month
    monthly_sales = df.groupby('Month')['Sales'].mean()
    sns.lineplot(
        ax=axes[1], x=monthly_sales.index.astype(str),
        y=monthly_sales.values
    )
    axes[1].set_title('Average Sales Per Month')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Average Sales')
    
    # Adjust layout and save plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.show()
    


def plot_categorical_plot(df):
    """
    Creates a bar chart showing total Sales per Category
    and a pie chart showing the proportion of Ship Mode.
    Saves the plots as 'categorical_plot.png'.
    """

    # Group data for category-wise sales
    category_sales = df.groupby('Category')['Sales'].sum().sort_values()
    
    # Count of each Ship Mode
    ship_mode_counts = df['Ship Mode'].value_counts()
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Bar plot: Total Sales by Category
    sns.barplot(ax=axes[0], x=category_sales.index, y=category_sales.values)
    axes[0].set_title('Total Sales by Category')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Total Sales')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

    # Pie chart: Distribution of Ship Mode
    axes[1].pie(ship_mode_counts, labels=ship_mode_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette('pastel'))
    axes[1].set_title('Ship Mode Distribution')
    
    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.show()
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Creates a correlation heatmap for numerical variables
    and a box plot for Sales by Region.
    Saves the plots as 'statistical_plot.png'.
    """

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    sns.heatmap(
        df.corr(numeric_only=True), annot=True,
        cmap='coolwarm', fmt='.2f', ax=axes[0]
    )
    axes[0].set_title('Correlation Heatmap')
    
    sns.boxplot(ax=axes[1], x=df['Region'], y=df['Sales'])
    axes[1].set_title('Sales Distribution by Region')
    axes[1].set_xlabel('Region')
    axes[1].set_ylabel('Sales')
    
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Computes the four main statistical moments for a given column.
    Returns mean, standard deviation, skewness, and excess kurtosis.
    """

    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col], nan_policy='omit')
    excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocess the dataset by handling missing values, 
    removing outliers from numerical columns using the IQR method,
    and displaying basic dataset information.
    """

    # Drop missing values
    df.dropna(inplace=True)
    
    # Identify numerical columns
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()

    # Remove outliers using IQR method
    for col in numerical_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    
    print(df.corr(numeric_only=True))
    
    return df


def writing(moments, col):
    """
    Prints the computed statistical moments with interpretation.
    """
    
    print(f'For the attribute {col}:')
    print(
        f'Mean = {moments[0]:.2f}, Standard Deviation = {moments[1]:.2f}, '
        f'Skewness = {moments[2]:.2f}, and Excess Kurtosis = {moments[3]:.2f}.'
    )
    if abs(moments[2]) < 0.5:
        skew_text = 'not skewed'
    elif moments[2] > 0:
        skew_text = 'right skewed'
    else:
        skew_text = 'left skewed'
    
    if moments[3] < -1:
        kurtosis_text = 'platykurtic'
    elif moments[3] > 1:
        kurtosis_text = 'leptokurtic'
    else:
        kurtosis_text = 'mesokurtic'
    
    print(f'The data is {skew_text} and {kurtosis_text}.')
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Sales'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
