import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics handling NaN values"""
    # Create mask for valid (non-NaN) values in both series
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    if np.sum(valid_mask) == 0:
        return {
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'Bias': np.nan,
            'R²': np.nan,
            'Valid_Points': 0
        }
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    
    # Handle division by zero in MAPE
    mape_mask = y_true_valid != 0
    if np.sum(mape_mask) > 0:
        mape = np.mean(np.abs((y_true_valid[mape_mask] - y_pred_valid[mape_mask]) / y_true_valid[mape_mask])) * 100
    else:
        mape = np.nan
    
    bias = np.mean(y_pred_valid - y_true_valid)
    
    # R² calculation with variance check
    if np.var(y_true_valid) > 1e-10:
        r2 = r2_score(y_true_valid, y_pred_valid)
    else:
        r2 = np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Bias': bias,
        'R²': r2,
        'Valid_Points': np.sum(valid_mask)
    }

def calculate_directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy - % of days where prediction and true value moved in same direction"""
    if len(y_true) < 2:
        return np.nan, 0
    
    # Calculate day-to-day changes
    true_changes = np.diff(y_true)
    pred_changes = np.diff(y_pred)
    
    # Create mask for valid changes (no NaN in either current or previous values)
    valid_mask = ~(np.isnan(true_changes) | np.isnan(pred_changes))
    
    if np.sum(valid_mask) == 0:
        return np.nan, 0
    
    true_changes_valid = true_changes[valid_mask]
    pred_changes_valid = pred_changes[valid_mask]
    
    # Check if changes have same sign (both positive, both negative, or both zero)
    same_direction = np.sign(true_changes_valid) == np.sign(pred_changes_valid)
    
    return np.mean(same_direction) * 100, np.sum(valid_mask)

def create_ensemble(df, weights):
    """Create weighted ensemble prediction handling NaN values"""
    w1, w2, w3 = weights
    
    # Initialize ensemble array
    ensemble = np.full(len(df), np.nan)
    
    # For each row, calculate ensemble using available models
    for i in range(len(df)):
        available_models = []
        available_weights = []
        
        # Check which models have valid predictions
        if not np.isnan(df['zGen3lastm'].iloc[i]):
            available_models.append(df['zGen3lastm'].iloc[i])
            available_weights.append(w1)
        
        if not np.isnan(df['zGen2'].iloc[i]):
            available_models.append(df['zGen2'].iloc[i])
            available_weights.append(w2)
        
        if not np.isnan(df['zGen3tf'].iloc[i]):
            available_models.append(df['zGen3tf'].iloc[i])
            available_weights.append(w3)
        
        # If we have at least one valid model, create ensemble
        if available_models:
            # Normalize weights to sum to 1 for available models
            available_weights = np.array(available_weights)
            if np.sum(available_weights) > 0:
                available_weights = available_weights / np.sum(available_weights)
                ensemble[i] = np.sum(np.array(available_models) * available_weights)
    
    return ensemble

def objective_function(weights, df):
    """Objective function for optimization (RMSE) handling NaN values"""
    ensemble = create_ensemble(df, weights)
    
    # Create mask for valid values in both target and ensemble
    valid_mask = ~(np.isnan(df['average']) | np.isnan(ensemble))
    
    if np.sum(valid_mask) == 0:
        return 1e6  # Return large penalty if no valid predictions
    
    y_true_valid = df['average'][valid_mask]
    y_pred_valid = ensemble[valid_mask]
    
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    return rmse

def optimize_weights(df):
    """Perform grid search and scipy optimization to find best weights"""
    print("Performing weight optimization...")
    
    # Constraints: weights sum to 1, all weights >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1), (0, 1), (0, 1)]
    
    # Try multiple starting points
    best_result = None
    best_rmse = float('inf')
    
    starting_points = [
        [0.6, 0.15, 0.25],  # Initial weights
        [0.5, 0.5, 0.0],  # No auxiliary
        [0.7, 0.2, 0.1],  # More base weight
        [0.4, 0.4, 0.2],  # More balanced
        [1.0, 0.0, 0.0],  # Only base model
    ]
    
    for start_weights in starting_points:
        try:
            result = minimize(
                objective_function,
                start_weights,
                args=(df,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'disp': False}
            )
            
            if result.success and result.fun < best_rmse:
                best_rmse = result.fun
                best_result = result
        except:
            continue
    
    # Also perform a coarse grid search
    grid_rmse = float('inf')
    grid_weights = None
    
    for w1 in np.arange(0.0, 1.01, 0.1):
        for w2 in np.arange(0.0, 1.01 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 >= 0:
                weights = [w1, w2, w3]
                rmse = objective_function(weights, df)
                if rmse < grid_rmse:
                    grid_rmse = rmse
                    grid_weights = weights
    
    # Return the best result
    if best_result is not None and best_rmse <= grid_rmse:
        return best_result.x, best_rmse
    else:
        return grid_weights, grid_rmse

def main():
    print("=== Time Series Ensemble Optimization ===\n")
    
    # Step 1: Read the CSV file
    try:
        # Try to read from uploaded file first
        df = pd.read_csv('your_file_modified.csv')  # Replace with actual filename
        print(f"Successfully loaded data with {len(df)} rows")
    except:
        print("Please upload your CSV file or update the filename in the script")
        return
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Step 2: Analyze missing data instead of dropping rows
    relevant_columns = ['average', 'zGen3', 'zGen3lastm', 'zGen3tf']
    
    print(f"\n=== MISSING DATA ANALYSIS ===")
    for col in relevant_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            print(f"{col}: {missing_count} missing ({missing_pct:.1f}%)")
        else:
            print(f"Warning: Column '{col}' not found in data")
    
    # Check for rows where target is available
    target_available = df['average'].notna().sum()
    print(f"\nRows with valid target ('average'): {target_available} out of {len(df)} ({target_available/len(df)*100:.1f}%)")
    
    if target_available == 0:
        print("Error: No valid target values found!")
        return
    
    # Step 3 & 4: Create ensemble with static weights (handling NaN)
    static_weights = [0.6, 0.15, 0.25]  # w1, w2, w3
    df['ensemble_static'] = create_ensemble(df, static_weights)
    
    # Step 5: Calculate metrics for static ensemble
    print(f"\n=== STATIC WEIGHTS RESULTS ===")
    print(f"Weights: zGen3lastm={static_weights[0]:.1f}, zGen2={static_weights[1]:.1f}, zGen3tf={static_weights[2]:.1f}")
    
    static_metrics = calculate_metrics(df['average'].values, df['ensemble_static'].values)
    static_dir_acc, static_dir_count = calculate_directional_accuracy(df['average'].values, df['ensemble_static'].values)
    
    print(f"\nStatic Ensemble Metrics:")
    for metric, value in static_metrics.items():
        if metric == 'Valid_Points':
            print(f"  {metric}: {value}")
        elif not np.isnan(value):
            print(f"  {metric}: {value:.6f}")
        else:
            print(f"  {metric}: N/A (insufficient data)")
    
    if not np.isnan(static_dir_acc):
        print(f"  Directional Accuracy: {static_dir_acc:.2f}% (based on {static_dir_count} valid transitions)")
    else:
        print(f"  Directional Accuracy: N/A (insufficient data)")
    
    # Calculate individual model metrics for comparison
    print(f"\n=== INDIVIDUAL MODEL PERFORMANCE ===")
    models = ['zGen3lastm', 'zGen2', 'zGen3tf']
    individual_metrics = {}
    
    for model in models:
        if model in df.columns:
            metrics = calculate_metrics(df['average'].values, df[model].values)
            dir_acc, dir_count = calculate_directional_accuracy(df['average'].values, df[model].values)
            individual_metrics[model] = {**metrics, 'Directional Accuracy': dir_acc, 'Dir_Count': dir_count}
            
            print(f"\n{model}:")
            for metric, value in metrics.items():
                if metric == 'Valid_Points':
                    print(f"  {metric}: {value}")
                elif not np.isnan(value):
                    print(f"  {metric}: {value:.6f}")
                else:
                    print(f"  {metric}: N/A")
            
            if not np.isnan(dir_acc):
                print(f"  Directional Accuracy: {dir_acc:.2f}% (based on {dir_count} transitions)")
            else:
                print(f"  Directional Accuracy: N/A")
        else:
            print(f"\n{model}: Column not found in data")
    
    # Step 6: Optimize weights
    print(f"\n=== WEIGHT OPTIMIZATION ===")
    try:
        optimal_weights, optimal_rmse = optimize_weights(df)
        df['ensemble_optimal'] = create_ensemble(df, optimal_weights)
        
        print(f"Optimal weights found:")
        print(f"  zGen3lastm: {optimal_weights[0]:.3f}")
        print(f"  zGen2: {optimal_weights[1]:.3f}")
        print(f"  zGen3tf: {optimal_weights[2]:.3f}")
        print(f"  Sum: {sum(optimal_weights):.3f}")
        
        # Calculate metrics for optimal ensemble
        optimal_metrics = calculate_metrics(df['average'].values, df['ensemble_optimal'].values)
        optimal_dir_acc, optimal_dir_count = calculate_directional_accuracy(df['average'].values, df['ensemble_optimal'].values)
        
        print(f"\nOptimal Ensemble Metrics:")
        for metric, value in optimal_metrics.items():
            if metric == 'Valid_Points':
                print(f"  {metric}: {value}")
            elif not np.isnan(value):
                print(f"  {metric}: {value:.6f}")
            else:
                print(f"  {metric}: N/A")
        
        if not np.isnan(optimal_dir_acc):
            print(f"  Directional Accuracy: {optimal_dir_acc:.2f}% (based on {optimal_dir_count} transitions)")
        else:
            print(f"  Directional Accuracy: N/A")
        
        # Compare improvements
        print(f"\n=== IMPROVEMENT ANALYSIS ===")
        if not np.isnan(static_metrics['RMSE']) and not np.isnan(optimal_metrics['RMSE']):
            rmse_improvement = ((static_metrics['RMSE'] - optimal_metrics['RMSE']) / static_metrics['RMSE']) * 100
            print(f"RMSE improvement: {rmse_improvement:.2f}%")
        else:
            print("RMSE improvement: Cannot calculate (insufficient data)")
        
        if not np.isnan(static_metrics['MAE']) and not np.isnan(optimal_metrics['MAE']):
            mae_improvement = ((static_metrics['MAE'] - optimal_metrics['MAE']) / static_metrics['MAE']) * 100
            print(f"MAE improvement: {mae_improvement:.2f}%")
        else:
            print("MAE improvement: Cannot calculate (insufficient data)")
        
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        optimal_weights = static_weights
        optimal_metrics = static_metrics
    
    # Step 7: Analysis and conclusions
    print(f"\n=== ENSEMBLE ANALYSIS ===")
    
    # Compare base model (zGen3lastm) with ensemble
    if 'zGen3lastm' in individual_metrics and not np.isnan(individual_metrics['zGen3lastm']['RMSE']):
        base_rmse = individual_metrics['zGen3lastm']['RMSE']
        ensemble_rmse = optimal_metrics.get('RMSE', static_metrics['RMSE'])
        
        if not np.isnan(ensemble_rmse):
            base_improvement = ((base_rmse - ensemble_rmse) / base_rmse) * 100
            print(f"Base model (zGen3lastm) RMSE: {base_rmse:.6f}")
            print(f"Optimal ensemble RMSE: {ensemble_rmse:.6f}")
            print(f"Improvement over base: {base_improvement:.2f}%")
        else:
            print("Cannot calculate improvement: insufficient ensemble data")
            base_improvement = 0
    else:
        print("Cannot analyze base model: insufficient zGen3lastm data")
        base_improvement = 0
    
    # Analyze component contributions
    print(f"\nComponent Analysis:")
    if optimal_weights[1] > 0.05:  # zGen3lastm has meaningful weight
        print(f"✓ zGen2 (variance learner) contributes meaningfully ({optimal_weights[1]:.1%})")
        if 'zGen2' in individual_metrics and 'zGen2' in individual_metrics:
            if not np.isnan(individual_metrics['zGen2']['RMSE']) and not np.isnan(individual_metrics['zGen3lastm']['RMSE']):
                variance_benefit = individual_metrics['zGen2']['RMSE'] < individual_metrics['zGen3lastm']['RMSE']
                print(f"  - Individual performance vs base: {'Better' if variance_benefit else 'Worse'}")
            else:
                print(f"  - Individual performance comparison: Insufficient data")
        print(f"  - Data availability: {individual_metrics.get('zGen2', {}).get('Valid_Points', 'Unknown')} valid points")
    else:
        print(f"✗ zGen2 has minimal contribution ({optimal_weights[1]:.1%})")
    
    if optimal_weights[2] > 0.05:  # zGen3tf has meaningful weight
        print(f"✓ zGen3tf (auxiliary) contributes meaningfully ({optimal_weights[2]:.1%})")
        if 'zGen3tf' in individual_metrics and 'zGen3' in individual_metrics:
            if not np.isnan(individual_metrics['zGen3tf']['RMSE']) and not np.isnan(individual_metrics['zGen3']['RMSE']):
                aux_benefit = individual_metrics['zGen3tf']['RMSE'] < individual_metrics['zGen3']['RMSE']
                print(f"  - Individual performance vs base: {'Better' if aux_benefit else 'Worse'}")
            else:
                print(f"  - Individual performance comparison: Insufficient data")
        print(f"  - Data availability: {individual_metrics.get('zGen3tf', {}).get('Valid_Points', 'Unknown')} valid points")
    else:
        print(f"✗ zGen3tf has minimal contribution ({optimal_weights[2]:.1%})")
    
    # Data coverage analysis
    print(f"\nData Coverage Analysis:")
    total_rows = len(df)
    target_coverage = static_metrics.get('Valid_Points', 0)
    print(f"Target variable coverage: {target_coverage}/{total_rows} ({target_coverage/total_rows*100:.1f}%)")
    
    for model in ['zGen3lastm', 'zGen2', 'zGen3tf']:
        if model in individual_metrics:
            model_coverage = individual_metrics[model].get('Valid_Points', 0)
            print(f"{model} coverage: {model_coverage}/{total_rows} ({model_coverage/total_rows*100:.1f}%)")
    
    # Final recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    if base_improvement > 1:
        print("✓ Ensemble approach provides meaningful improvement over base model")
    elif base_improvement > 0:
        print("~ Ensemble provides modest improvement over base model")
    else:
        print("✗ Ensemble does not improve over base model - consider using zGen3lastm alone")
    
    if optimal_weights[1] > 0.1 and optimal_weights[2] > 0.1:
        print("✓ Both auxiliary models contribute to ensemble performance")
    elif optimal_weights[1] > 0.1:
        print("✓ Variance learner (zGen2) is the primary beneficial auxiliary model")
    elif optimal_weights[2] > 0.1:
        print("✓ Auxiliary model (zGen3tf) is the primary beneficial auxiliary model")
    else:
        print("~ Consider focusing on the base model (zGen3lastm) with minimal ensemble weighting")
    
    # Handle missing data recommendations
    missing_data_issues = []
    for model in ['zGen3lastm', 'zGen2', 'zGen3tf']:
        if model in individual_metrics:
            coverage = individual_metrics[model].get('Valid_Points', 0)
            if coverage < target_coverage * 0.8:  # Less than 80% of target coverage
                missing_data_issues.append(model)
    
    if missing_data_issues:
        print(f"\n⚠️  Missing Data Concerns:")
        for model in missing_data_issues:
            print(f"   - {model} has significant missing data, limiting its contribution")
        print("   - Consider improving data collection or imputation methods for these models")

if __name__ == "__main__":
    main()