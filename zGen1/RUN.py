from zGen1.Deepalgo import RealizedVolatilityPipeline
import pandas as pd
data = pd.read_csv('data/SPY.csv')

# Initialize pipeline
pipeline = RealizedVolatilityPipeline()

print(f"Raw data length: {len(data)}")
print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
# Run analysis
results = pipeline.run_pipeline(
    data_path='data/SPY.csv',
    output_dir='output/spy_analysis'
)

# View results
print(f"Test R²: {results['forecast_results']['test_r2']:.4f}")