import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')


class AirlinePassengerEDA:
    def __init__(self, file_path):
        """Initialize with the CSV file path"""
        self.df = self._load_data(file_path)

    def _load_data(self, file_path):
        """Load and preprocess the data"""
        df = pd.read_csv(
            file_path,
            sep='\t',
            parse_dates={'date': ['Year', 'Month']},
            date_parser=lambda x: pd.to_datetime(x, format='%Y %m'),
        )

        # Create derived features
        df['total_pax'] = df['Pax From Origin'] + df['Pax To Origin']
        df['pax_difference'] = df['Pax From Origin'] - df['Pax To Origin']
        df['route'] = df['Origin'] + '_' + df['Dest']

        # Add temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        return df

    def temporal_analysis(self):
        """Perform temporal analysis of passenger traffic"""
        # Monthly aggregations
        monthly_stats = self.df.groupby('date').agg(
            {
                'total_pax': ['mean', 'std', 'min', 'max'],
                'Pax From Origin': 'sum',
                'Pax To Origin': 'sum',
            }
        )

        # Seasonal decomposition
        decomposition = seasonal_decompose(
            monthly_stats['total_pax']['mean'], period=12, extrapolate_trend='freq'
        )

        # Plot temporal patterns
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))

        # Original data
        monthly_stats['total_pax']['mean'].plot(ax=axes[0])
        axes[0].set_title('Monthly Average Total Passengers')

        # Trend
        decomposition.trend.plot(ax=axes[1])
        axes[1].set_title('Trend')

        # Seasonal
        decomposition.seasonal.plot(ax=axes[2])
        axes[2].set_title('Seasonal Pattern')

        # Residual
        decomposition.resid.plot(ax=axes[3])
        axes[3].set_title('Residuals')

        plt.tight_layout()
        plt.savefig('charts/temporal_analysis.png', bbox_inches='tight')

        return fig, monthly_stats, decomposition

    def distribution_analysis(self):
        """Analyze distribution of passenger numbers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Total passengers distribution
        sns.histplot(self.df['total_pax'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of Total Passengers')

        # QQ plot
        stats.probplot(self.df['total_pax'], dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Total Passengers')

        # Box plot by month
        sns.boxplot(x='month', y='total_pax', data=self.df, ax=axes[1, 0])
        axes[1, 0].set_title('Monthly Distribution')

        # Box plot by route
        sns.boxplot(x='route', y='total_pax', data=self.df, ax=axes[1, 1])
        axes[1, 1].set_title('Route Distribution')
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Calculate summary statistics
        stats_summary = {
            'mean': self.df['total_pax'].mean(),
            'median': self.df['total_pax'].median(),
            'std': self.df['total_pax'].std(),
            'skew': self.df['total_pax'].skew(),
            'kurtosis': self.df['total_pax'].kurtosis(),
        }

        print('stats_summary::: ')
        print(stats_summary)

        plt.savefig('charts/distribution_analysis.png', bbox_inches='tight')

        return fig, stats_summary

    def correlation_analysis(self):
        """Analyze correlations between features"""
        # Create correlation matrix
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()

        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True
        )
        plt.title('Correlation Heatmap')

        plt.savefig('charts/correlation_heatmap.png', bbox_inches='tight')

        return plt.gcf(), corr_matrix

    def anomaly_detection(self, threshold=3):
        """Detect anomalies using Z-score method"""
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(self.df['total_pax']))

        # Identify anomalies
        anomalies = self.df[z_scores > threshold].copy()

        # Plot anomalies
        plt.figure(figsize=(15, 6))
        plt.plot(self.df['date'], self.df['total_pax'], label='Normal')
        plt.scatter(
            anomalies['date'], anomalies['total_pax'], color='red', label='Anomaly'
        )
        plt.title('Passenger Traffic with Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Total Passengers')
        plt.legend()

        plt.savefig('charts/anomaly_detection.png', bbox_inches='tight')

        return plt.gcf(), anomalies

    def route_analysis(self):
        """Analyze route-specific patterns"""
        # Route summary
        route_summary = (
            self.df.groupby('route')
            .agg(
                {
                    'total_pax': ['mean', 'std', 'count'],
                    'pax_difference': ['mean', 'std'],
                }
            )
            .round(2)
        )

        # Plot route comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='route', y='total_pax', data=self.df)
        plt.title('Average Passengers by Route')
        plt.xticks(rotation=45)

        plt.savefig('charts/route_analysis.png', bbox_inches='tight')

        return plt.gcf(), route_summary

    def generate_full_report(self):
        """Generate a complete EDA report"""
        report = {}

        # Temporal analysis
        report['temporal'] = self.temporal_analysis()

        # Distribution analysis
        report['distribution'] = self.distribution_analysis()

        # Correlation analysis
        report['correlation'] = self.correlation_analysis()

        # Anomaly detection
        report['anomalies'] = self.anomaly_detection()

        # Route analysis
        report['route'] = self.route_analysis()

        return report


def main():
    # Initialize EDA
    eda = AirlinePassengerEDA('./data/passengers_1_blr_delhi.csv')

    # Generate full report
    report = eda.generate_full_report()

    # Display some key findings
    print("\nKey Statistics:")
    print("--------------")
    print(f"Total Routes: {len(eda.df['route'].unique())}")
    print(f"Date Range: {eda.df['date'].min()} to {eda.df['date'].max()}")
    print(f"Average Daily Passengers: {eda.df['total_pax'].mean():,.0f}")
    print(f"Busiest Route: {eda.df.groupby('route')['total_pax'].mean().idxmax()}")

    return eda, report


if __name__ == "__main__":
    eda, report = main()
