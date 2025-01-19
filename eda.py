from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose


class PassengerDemandEDA:
    def __init__(self, df):
        """
        Initialize with a DataFrame containing passenger demand data
        Expected columns: date, passengers, other features like weather, events, etc.
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)

    def temporal_analysis(self):
        """Analyze temporal patterns in passenger demand"""
        # Resample data at different frequencies
        daily_avg = self.df['passengers'].resample('D').mean()
        weekly_avg = self.df['passengers'].resample('W').mean()
        monthly_avg = self.df['passengers'].resample('ME').mean()

        # Seasonal decomposition
        decomposition = seasonal_decompose(self.df['passengers'], period=6)
        decomposition.plot()

        return {
            'daily_avg': daily_avg,
            'weekly_avg': weekly_avg,
            'monthly_avg': monthly_avg,
            'seasonal_decomp': decomposition,
        }

    def distribution_analysis(self):
        """Analyze the distribution of passenger numbers"""
        stats_dict = {
            'mean': self.df['passengers'].mean(),
            'median': self.df['passengers'].median(),
            'std': self.df['passengers'].std(),
            'skew': self.df['passengers'].skew(),
            'kurtosis': self.df['passengers'].kurtosis(),
            'q1': self.df['passengers'].quantile(0.25),
            'q3': self.df['passengers'].quantile(0.75),
        }

        return stats_dict

    def correlation_analysis(self):
        """Analyze correlations between features"""
        # Calculate correlation matrix
        corr_matrix = self.df.corr()

        # Find top correlations with passenger numbers
        passenger_corr = corr_matrix['passengers'].sort_values(ascending=False)

        return {
            'correlation_matrix': corr_matrix,
            'passenger_correlations': passenger_corr,
        }

    def anomaly_detection(self):
        """Detect anomalies in passenger numbers"""
        # Z-score based anomaly detection
        z_scores = np.abs(stats.zscore(self.df['passengers']))
        anomalies = self.df[z_scores > 3].copy()

        return anomalies

    def plot_temporal_patterns(self):
        """Plot various temporal patterns"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Time series plot
        self.df['passengers'].plot(ax=axes[0])
        axes[0].set_title('Passenger Demand Over Time')

        # Monthly box plot
        self.df['month'] = self.df.index.month
        sns.boxplot(data=self.df, x='month', y='passengers', ax=axes[1])
        axes[1].set_title('Monthly Distribution of Passenger Demand')

        # Weekly patterns
        self.df['dayofweek'] = self.df.index.dayofweek
        sns.boxplot(data=self.df, x='dayofweek', y='passengers', ax=axes[2])
        axes[2].set_title('Day of Week Distribution of Passenger Demand')

        plt.tight_layout()
        plt.show()
        return fig

    def generate_report(self):
        """Generate a comprehensive EDA report"""
        temporal = self.temporal_analysis()
        distribution = self.distribution_analysis()
        correlations = self.correlation_analysis()
        anomalies = self.anomaly_detection()
        plt = self.plot_temporal_patterns()

        report = {
            'temporal_analysis': temporal,
            'distribution_statistics': distribution,
            'correlations': correlations,
            'anomalies': anomalies,
        }

        return report


# Example usage function
def main():
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='ME')
    np.random.seed(42)

    # BENGALURU -> DELHI
    passengers = np.array(
        [
            206890,
            189950,
            202622,
            211000,
            192506,
            184262,
            196823,
            209179,
            191576,
            215227,
            188673,
            204471,
        ]
    )
    weather_temp = np.random.normal(20, 5, 12)
    special_events = np.random.choice([0, 1], 12, p=[0.9, 0.1])

    """
    # Original code

    # Generate synthetic passenger data with seasonal patterns and trends
    passengers = np.random.normal(1000, 100, len(dates))
    passengers += np.sin(np.linspace(0, 4*np.pi, len(dates))) * 200  # Seasonal pattern
    passengers += np.linspace(0, 300, len(dates))  # Upward trend
    
    # Create sample weather and event data
    weather_temp = np.random.normal(20, 5, len(dates))
    special_events = np.random.choice([0, 1], len(dates), p=[0.9, 0.1])
    """

    # Create DataFrame
    df = pd.DataFrame(
        {
            'date': dates,
            'passengers': passengers,
            'temperature': weather_temp,
            'special_events': special_events,
        }
    )

    # # Initialize EDA class
    eda = PassengerDemandEDA(df)

    # # Generate report
    report = eda.generate_report()


# Main
main()
