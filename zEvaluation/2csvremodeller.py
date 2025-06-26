import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import re
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class VolatilityAnalyzer:
    def __init__(self, true_values_file: str = None, predictions_folder: str = None, merged_data: pd.DataFrame = None):
        """
        Initialize the analyzer with paths to true values file and predictions folder,
        or directly with already merged data.

        Args:
            true_values_file: Path to CSV file with true volatility values
            predictions_folder: Path to folder containing daily prediction files
            merged_data: (Optional) Already merged DataFrame
        """
        self.true_values_file = true_values_file
        self.predictions_folder = predictions_folder
        self.merged_data = merged_data
        self.error_metrics = None
        
    def load_true_values(self) -> pd.DataFrame:
        """Load the true values CSV file"""
        try:
            df = pd.read_csv(self.true_values_file)
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"Loaded true values: {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading true values file: {e}")
            return None
    
    def extract_date_from_filename(self, filename: str) -> str:
        """Extract date from filename (format: YYYY-MM-DD)"""
        # Look for date pattern YYYY-MM-DD in filename
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        match = re.search(date_pattern, filename)
        if match:
            return match.group(1)
        return None
    
    def load_predictions(self) -> Dict[str, pd.DataFrame]:
        """
        Load all prediction files from the folder
        
        Returns:
            Dictionary with date as key and DataFrame as value
        """
        predictions = {}
        
        # Get all CSV files in the predictions folder
        csv_files = glob.glob(os.path.join(self.predictions_folder, "*.csv"))
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            date_str = self.extract_date_from_filename(filename)
            
            if date_str:
                try:
                    df = pd.read_csv(file_path)
                    predictions[date_str] = df
                    print(f"Loaded predictions for {date_str}: {len(df)} records")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Could not extract date from filename: {filename}")
        
        return predictions
    
    def merge_data(self) -> pd.DataFrame:
        """
        Merge true values with predictions
        
        Returns:
            Merged DataFrame with true values and predictions
        """
        true_df = self.load_true_values()
        if true_df is None:
            return None
        
        predictions = self.load_predictions()
        if not predictions:
            print("No predictions loaded")
            return None
        
        # Start with true values DataFrame
        merged_df = true_df.copy()
        
        # Get all unique descriptions from all prediction files
        all_descriptions = set()
        for pred_df in predictions.values():
            if 'description' in pred_df.columns:
                all_descriptions.update(pred_df['description'].unique())
        
        # Initialize columns for each description
        for desc in all_descriptions:
            merged_df[desc] = np.nan
        
        # Fill in prediction values
        for date_str, pred_df in predictions.items():
            try:
                date = pd.to_datetime(date_str)
                mask = merged_df['Date'] == date
                
                if mask.any():
                    for _, row in pred_df.iterrows():
                        if 'description' in row and 'annualized_volatility' in row:
                            desc = row['description']
                            vol = row['annualized_volatility']
                            
                            # Convert percentage to decimal if needed (assuming true values are in decimal form)
                            if vol > 1:  # Likely percentage
                                vol = vol / 100
                            
                            merged_df.loc[mask, desc] = vol
                else:
                    print(f"No matching date found for {date_str}")
            except Exception as e:
                print(f"Error processing predictions for {date_str}: {e}")
        
        merged_df['ensemble'] = merged_df['zGen3lastm'] * 0.77 + merged_df["zGen2"] * 0.23
        self.merged_data = merged_df

        return merged_df
    
    def calculate_error_metrics(self, true_col: str = 'average') -> pd.DataFrame:
        """
        Calculate comprehensive error metrics for predictions vs true values
        
        Args:
            true_col: Column name containing true values (default: 'average')
        
        Returns:
            DataFrame with error metrics for each prediction method
        """
        if self.merged_data is None:
            print("No merged data available. Run merge_data() first.")
            return None
        
        # Get prediction columns (exclude the known true value columns)
        true_value_cols = ['Date', 'RV_Parkinson', 'RV_GarmanKlass', 'RV_RogersSatchell', 
                          'RV_CloseToClose', 'average']
        pred_cols = [col for col in self.merged_data.columns if col not in true_value_cols]
        
        if not pred_cols:
            print("No prediction columns found")
            return None
        
        error_metrics = []
        
        for pred_col in pred_cols:
            # Remove rows with NaN values for this prediction
            valid_mask = (~self.merged_data[pred_col].isna()) & (~self.merged_data[true_col].isna())
            
            if valid_mask.sum() == 0:
                print(f"No valid data points for {pred_col}")
                continue
            
            y_true = self.merged_data.loc[valid_mask, true_col]
            y_pred = self.merged_data.loc[valid_mask, pred_col]
            
            # Calculate various error metrics
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Additional metrics
            bias = np.mean(y_pred - y_true)
            std_error = np.std(y_pred - y_true)
            r_squared = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            
            # Correlation coefficient
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            
            # Directional accuracy (for volatility, this might be less relevant)
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            directional_accuracy = np.mean(direction_true == direction_pred) * 100 if len(direction_true) > 0 else np.nan
            
            error_metrics.append({
                'Model': pred_col,
                'Data_Points': valid_mask.sum(),
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE (%)': mape,
                'Bias': bias,
                'Std_Error': std_error,
                'R_Squared': r_squared,
                'Correlation': correlation,
                'Directional_Accuracy (%)': directional_accuracy,
                'Min_Error': np.min(y_pred - y_true),
                'Max_Error': np.max(y_pred - y_true),
                'Q25_Error': np.percentile(y_pred - y_true, 25),
                'Q75_Error': np.percentile(y_pred - y_true, 75)
            })
        
        self.error_metrics = pd.DataFrame(error_metrics)
        return self.error_metrics
    
    def save_results(self, merged_file: str = 'merged_volatility_data.csv', 
                    metrics_file: str = 'error_metrics.csv'):
        """Save merged data and error metrics to CSV files"""
        if self.merged_data is not None:
            self.merged_data.to_csv(merged_file, index=False)
            print(f"Merged data saved to {merged_file}")
        
        if self.error_metrics is not None:
            self.error_metrics.to_csv(metrics_file, index=False)
            print(f"Error metrics saved to {metrics_file}")
    
    def plot_error_analysis(self, save_plots: bool = True):
        """Create visualizations for error analysis"""
        if self.error_metrics is None:
            print("No error metrics available. Run calculate_error_metrics() first.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. RMSE comparison
        axes[0, 0].bar(self.error_metrics['Model'], self.error_metrics['RMSE'])
        axes[0, 0].set_title('Root Mean Square Error by Model')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. R-squared comparison
        axes[0, 1].bar(self.error_metrics['Model'], self.error_metrics['R_Squared'])
        axes[0, 1].set_title('R-Squared by Model')
        axes[0, 1].set_ylabel('R-Squared')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. MAPE comparison
        axes[1, 0].bar(self.error_metrics['Model'], self.error_metrics['MAPE (%)'])
        axes[1, 0].set_title('Mean Absolute Percentage Error by Model')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Bias comparison
        axes[1, 1].bar(self.error_metrics['Model'], self.error_metrics['Bias'])
        axes[1, 1].set_title('Bias by Model')
        axes[1, 1].set_ylabel('Bias')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('error_analysis_plots.png', dpi=300, bbox_inches='tight')
            print("Error analysis plots saved to error_analysis_plots.png")
        
        plt.show()
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        if self.error_metrics is None:
            return "No error metrics available."
        
        report = []
        report.append("=" * 60)
        report.append("VOLATILITY PREDICTION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Best models by different metrics
        best_rmse = self.error_metrics.loc[self.error_metrics['RMSE'].idxmin()]
        best_r2 = self.error_metrics.loc[self.error_metrics['R_Squared'].idxmax()]
        best_mape = self.error_metrics.loc[self.error_metrics['MAPE (%)'].idxmin()]
        
        report.append("BEST PERFORMING MODELS:")
        report.append(f"• Lowest RMSE: {best_rmse['Model']} (RMSE: {best_rmse['RMSE']:.6f})")
        report.append(f"• Highest R²: {best_r2['Model']} (R²: {best_r2['R_Squared']:.4f})")
        report.append(f"• Lowest MAPE: {best_mape['Model']} (MAPE: {best_mape['MAPE (%)']:.2f}%)")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append(f"• Number of models evaluated: {len(self.error_metrics)}")
        report.append(f"• Average RMSE across models: {self.error_metrics['RMSE'].mean():.6f}")
        report.append(f"• Average R² across models: {self.error_metrics['R_Squared'].mean():.4f}")
        report.append(f"• Average MAPE across models: {self.error_metrics['MAPE (%)'].mean():.2f}%")
        report.append("")
        
        # Detailed metrics table
        report.append("DETAILED METRICS:")
        report.append(self.error_metrics.to_string(index=False))
        
        return "\n".join(report)

# Usage example and main execution
def main():
    """Main function to run the analysis"""
    # Option 1: Use merged data directly
    merged_data_file = "zEvaluation\merged_volatility_data.csv"  # e.g., "merged_volatility_data.csv" or set to None to use files
    if merged_data_file:
        merged_data = pd.read_csv(merged_data_file)
        merged_data['Date'] = pd.to_datetime(merged_data['Date'])
        analyzer = VolatilityAnalyzer(merged_data=merged_data)
    else:
        # Option 2: Merge from raw files
        true_values_file = "realized_volatility.csv"
        predictions_folder = "slicerresults"
        analyzer = VolatilityAnalyzer(true_values_file, predictions_folder)

    # Step 1: Merge data (skip if already provided)
    if analyzer.merged_data is None:
        print("Step 1: Merging data...")
        merged_data = analyzer.merge_data()
    else:
        merged_data = analyzer.merged_data
    if merged_data is not None:
        print(f"Successfully merged data. Shape: {merged_data.shape}")
        print("\nColumns in merged data:")
        print(merged_data.columns.tolist())

        # Step 2: Calculate error metrics
        print("\nStep 2: Calculating error metrics...")
        error_metrics = analyzer.calculate_error_metrics()

        if error_metrics is not None:
            print(f"Calculated metrics for {len(error_metrics)} models")

            # Step 3: Save results
            print("\nStep 3: Saving results...")
            analyzer.save_results()

            # Step 4: Generate plots
            print("\nStep 4: Generating visualizations...")
            analyzer.plot_error_analysis()

            # Step 5: Generate summary report
            print("\nStep 5: Generating summary report...")
            report = analyzer.generate_summary_report()
            print(report)

            # Save report to file
            with open('volatility_analysis_report.txt', 'w') as f:
                f.write(report)
            print("\nSummary report saved to volatility_analysis_report.txt")

        else:
            print("Failed to calculate error metrics")
    else:
        print("Failed to merge data")

if __name__ == "__main__":
    main()