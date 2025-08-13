"""
Example usage script for Semi-Parametric Counterfactual Demand Model
Updated to work with base prices and price ratios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from semi_parametric_model import SemiParametricDemandModel, ModelConfig
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_products=200, n_periods=50):
    """Generate synthetic data for testing with base prices"""
    
    np.random.seed(42)
    
    # Define KAN5 categories (single level) - expanded for more variety
    kan5_categories = [
        'KAN5_FRESH_FRUITS', 'KAN5_FRESH_VEGETABLES', 'KAN5_DAIRY', 'KAN5_BAKERY', 
        'KAN5_MEAT', 'KAN5_SEAFOOD', 'KAN5_FROZEN', 'KAN5_BEVERAGES', 
        'KAN5_SNACKS', 'KAN5_HOUSEHOLD'
    ]
    
    data = []
    
    for i in range(n_products * n_periods):
        sku_id = f'SKU_{i % n_products:04d}'
        kan5 = np.random.choice(kan5_categories)
        
        # Base elasticity varies by KAN5 category - realistic values
        if 'FRESH' in kan5:
            base_elasticity = -2.5 + np.random.normal(0, 0.2)  # Fresh items are highly elastic
        elif 'DAIRY' in kan5:
            base_elasticity = -1.8 + np.random.normal(0, 0.15)
        elif 'MEAT' in kan5 or 'SEAFOOD' in kan5:
            base_elasticity = -1.2 + np.random.normal(0, 0.1)  # Less elastic
        elif 'BEVERAGES' in kan5:
            base_elasticity = -1.6 + np.random.normal(0, 0.2)
        elif 'HOUSEHOLD' in kan5:
            base_elasticity = -0.8 + np.random.normal(0, 0.1)  # Necessities, less elastic
        elif 'SNACKS' in kan5:
            base_elasticity = -2.0 + np.random.normal(0, 0.2)  # Impulse purchases, variable
        else:
            base_elasticity = -1.5 + np.random.normal(0, 0.2)
        
        # Generate base price for each SKU
        if 'FRESH' in kan5:
            base_price = np.random.uniform(2, 10)  # Lower unit prices
        elif 'MEAT' in kan5 or 'SEAFOOD' in kan5:
            base_price = np.random.uniform(10, 30)  # Higher unit prices
        else:
            base_price = np.random.uniform(5, 20)
        
        # Generate price ratios (actual_price / base_price)
        price_ratio = np.random.uniform(0.5, 1.2)  # 50% to 120% of base price
        actual_price = base_price * price_ratio
        
        # Sales volume varies by category
        if 'HOUSEHOLD' in kan5:
            base_sales_volume = np.random.uniform(20, 80)  # Lower volume
        elif 'FRESH' in kan5:
            base_sales_volume = np.random.uniform(100, 300)  # Higher volume, perishable
        else:
            base_sales_volume = np.random.uniform(50, 200)
        
        # Simulate sales with elasticity model
        # Higher price ratio = higher price = lower demand (with negative elasticity)
        price_effect = base_elasticity * np.log(price_ratio)
        
        # Add seasonality and noise
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / (n_periods * 7))  # Weekly seasonality
        noise_factor = np.random.uniform(0.7, 1.3)
        
        sales = base_sales_volume * np.exp(price_effect) * seasonal_factor * noise_factor
        
        # Time features
        day_of_week = i % 7
        week_of_month = (i // 7) % 4
        month = (i // 30) % 12
        is_weekend = 1 if day_of_week >= 5 else 0
        
        data.append({
            'sku_id': sku_id,
            'kan5': kan5,
            'base_price': base_price,
            'actual_price': actual_price,
            'price_ratio': price_ratio,
            'sales': max(0, sales),
            'day_of_week': day_of_week,
            'week_of_month': week_of_month,
            'month': month,
            'is_weekend': is_weekend,
            'inventory': np.random.uniform(50, 1000),
            'days_to_expiry': np.random.uniform(1, 30),
            'historical_avg_sales': base_sales_volume * np.random.uniform(0.8, 1.2),
            'sales_trend': np.random.uniform(-0.2, 0.2),
            'competitor_avg_price_ratio': price_ratio * np.random.uniform(0.9, 1.1),
            'sales_lag_1': base_sales_volume * np.random.uniform(0.7, 1.3),
            'sales_lag_7': base_sales_volume * np.random.uniform(0.6, 1.4),
            'sales_lag_14': base_sales_volume * np.random.uniform(0.5, 1.5)
        })
    
    return pd.DataFrame(data)


def train_model():
    """Train the semi-parametric model"""
    logger.info("="*60)
    logger.info("Training Semi-Parametric Counterfactual Demand Model")
    logger.info("="*60)
    
    # Load config
    config = ModelConfig.from_config_file()
    logger.info(f"Loaded config with validation split: {config.validation_split}")
    
    # Generate synthetic data
    data = generate_synthetic_data(n_products=200, n_periods=50)
    logger.info(f"Generated {len(data)} training samples")
    logger.info(f"Price ratio range: {data['price_ratio'].min():.2f} - {data['price_ratio'].max():.2f}")
    
    # Initialize and train model
    model = SemiParametricDemandModel(config)
    model.fit(data)
    
    # Display results
    logger.info("\nTraining Metrics:")
    for metric, value in model.training_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    if model.validation_metrics:
        logger.info("\nValidation Metrics:")
        for metric, value in model.validation_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Display elasticities by KAN5 category
    logger.info("\nEstimated Price Elasticities by KAN5 Category:")
    for kan5 in sorted(data['kan5'].unique()):
        elasticity = model.get_elasticity(kan5)
        logger.info(f"  {kan5}: {elasticity:.3f}")
    
    # Save model using pickle
    model.save_model('trained_model.pkl')
    
    return model, data


def demonstrate_counterfactual_predictions(model, data):
    """Demonstrate counterfactual demand predictions"""
    logger.info("\n" + "="*60)
    logger.info("Counterfactual Demand Predictions")
    logger.info("="*60)
    
    # Select sample products
    sample_products = data.groupby(['kan5']).first().reset_index().head(4)
    
    # Define price ratio scenarios (0.5 = 50% of base price, 1.0 = base price)
    price_ratio_scenarios = np.linspace(0.5, 1.2, 20)
    
    # Create subplot for each product
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (_, product) in enumerate(sample_products.iterrows()):
        # Get base sales and base price ratio for this SKU
        sku_data = data[data['sku_id'] == product['sku_id']]
        base_price_ratio = sku_data['price_ratio'].mean()
        base_sales = sku_data['sales'].mean()
        
        # Predict demand curve
        price_ratios, demands = model.predict_demand_curve(
            base_price_ratio=base_price_ratio,
            base_sales=base_sales,
            kan5=product['kan5'],
            price_ratio_range=price_ratio_scenarios
        )
        
        # Calculate revenue curve
        base_price = product['base_price']
        actual_prices = base_price * price_ratios
        revenues = demands * actual_prices
        
        # Plot demand curve
        ax = axes[idx]
        ax2 = ax.twinx()
        
        # Convert to discount percentage for display
        discount_percentages = (1 - price_ratios) * 100
        
        # Demand line
        line1 = ax.plot(discount_percentages, demands, 'b-', linewidth=2, label='Demand')
        ax.set_xlabel('Discount from Base Price (%)')
        ax.set_ylabel('Demand (units)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Revenue line
        line2 = ax2.plot(discount_percentages, revenues, 'g--', linewidth=2, label='Revenue')
        ax2.set_ylabel('Revenue ($)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Title and legend
        ax.set_title(f'{product["kan5"]} (Base Price: ${base_price:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        # Log predictions at key price points
        logger.info(f"\n{product['kan5']}:")
        test_price_ratios = [1.0, 0.9, 0.7, 0.5]  # 100%, 90%, 70%, 50% of base price
        for pr in test_price_ratios:
            demand = model.predict_counterfactual_demand(
                target_price_ratio=pr,
                base_price_ratio=base_price_ratio,
                base_sales=base_sales,
                kan5=product['kan5']
            )
            revenue = demand * base_price * pr
            discount_pct = (1 - pr) * 100
            logger.info(f"  {discount_pct:.0f}% off (ratio={pr:.1f}): Demand={demand:.1f}, Revenue=${revenue:.2f}")
    
    plt.suptitle('Counterfactual Demand and Revenue Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig('counterfactual_curves.png', dpi=150, bbox_inches='tight')
    logger.info("\nSaved demand curves to counterfactual_curves.png")


def create_sku_price_sales_curves(model, data):
    """Create price-sales curves for individual SKUs"""
    logger.info("\n" + "="*60)
    logger.info("Creating Individual SKU Price-Sales Curves")
    logger.info("="*60)
    
    # Select diverse SKUs from different categories
    target_categories = ['KAN5_FRESH_FRUITS', 'KAN5_DAIRY', 'KAN5_HOUSEHOLD', 'KAN5_MEAT']
    sample_skus = []
    
    for kan5 in target_categories:
        category_data = data[data['kan5'] == kan5]
        if len(category_data) > 0:
            # Get first SKU from category for consistency
            sku_id = category_data['sku_id'].iloc[0]
            sku_data = data[data['sku_id'] == sku_id].iloc[0]
            sample_skus.append(sku_data)
            logger.info(f"Selected SKU {sku_id} from {kan5}")
    
    # Create price range
    price_ratios = np.linspace(0.5, 1.2, 15)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, sku_data in enumerate(sample_skus):
        sku_id = sku_data['sku_id']
        kan5 = sku_data['kan5']
        base_price = sku_data['base_price']
        
        # Get historical data for this SKU
        sku_historical = data[data['sku_id'] == sku_id]
        base_price_ratio = sku_historical['price_ratio'].mean()
        base_sales = sku_historical['sales'].mean()
        
        # Predict demand at different price ratios
        demands = []
        for pr in price_ratios:
            demand = model.predict_counterfactual_demand(
                target_price_ratio=pr,
                base_price_ratio=base_price_ratio,
                base_sales=base_sales,
                kan5=kan5
            )
            demands.append(demand)
        
        demands = np.array(demands)
        actual_prices = price_ratios * base_price
        
        # Plot the predicted curve
        plt.plot(actual_prices, demands, 
                marker='o', markersize=4, linewidth=2, 
                color=colors[idx], 
                label=f"{sku_id} ({kan5.replace('KAN5_', '')})",
                zorder=3)
        
        # Add historical points
        hist_prices = sku_historical['actual_price'].values
        hist_sales = sku_historical['sales'].values
        plt.scatter(hist_prices, hist_sales, 
                   s=20, alpha=0.5, color=colors[idx], 
                   marker='s', zorder=2)
        
        # Log details
        logger.info(f"\n{sku_id} ({kan5}):")
        logger.info(f"  Elasticity: {model.get_elasticity(kan5):.3f}")
        logger.info(f"  Base price: ${base_price:.2f}")
        logger.info(f"  Predicted demand range: {demands.min():.1f} - {demands.max():.1f} units")
    
    # Formatting
    plt.xlabel('Price ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Sales (units)', fontsize=12, fontweight='bold')
    plt.title('Price-Sales Curves for Individual SKUs\n'
              'Lines: Model predictions | Squares: Historical observations', 
              fontsize=14, fontweight='bold')
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Invert x-axis to show high price on left
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('sku_price_sales_curves.png', dpi=150, bbox_inches='tight')
    logger.info("\nSaved individual SKU price-sales curves to sku_price_sales_curves.png")


def test_model_persistence(model):
    """Test model save/load functionality"""
    logger.info("\n" + "="*60)
    logger.info("Testing Model Persistence")
    logger.info("="*60)
    
    # Load model in new instance
    new_model = SemiParametricDemandModel()
    new_model.load_model('trained_model.pkl')
    
    # Verify parameters match
    if model.theta is not None and new_model.theta is not None:
        if np.allclose(model.theta, new_model.theta):
            logger.info("✓ Model parameters successfully saved and loaded")
        else:
            logger.info("✗ Model parameters mismatch after loading")
    else:
        logger.info("✗ Model parameters not found")
    
    return new_model


def predict_on_new_data(model):
    """Example of using trained model on new data"""
    logger.info("\n" + "="*60)
    logger.info("Prediction on New Data")
    logger.info("="*60)
    
    # Create new product data with base prices
    new_products = pd.DataFrame([
        {'sku_id': 'NEW_001', 'kan5': 'KAN5_FRESH_FRUITS', 
         'base_price': 5.99, 'current_price_ratio': 0.9, 'current_sales': 150},
        {'sku_id': 'NEW_002', 'kan5': 'KAN5_DAIRY',
         'base_price': 3.49, 'current_price_ratio': 0.85, 'current_sales': 100},
        {'sku_id': 'NEW_003', 'kan5': 'KAN5_FRESH_VEGETABLES',
         'base_price': 2.99, 'current_price_ratio': 0.95, 'current_sales': 200},
    ])
    
    # Predict at various price ratios
    target_price_ratios = [1.0, 0.9, 0.8, 0.7, 0.6]
    
    results = []
    for _, product in new_products.iterrows():
        logger.info(f"\n{product['sku_id']} ({product['kan5']}):")
        logger.info(f"  Base price: ${product['base_price']:.2f}")
        logger.info(f"  Current: {(1-product['current_price_ratio'])*100:.0f}% off, {product['current_sales']:.0f} units")
        logger.info("  Predictions:")
        
        for target_ratio in target_price_ratios:
            demand = model.predict_counterfactual_demand(
                target_price_ratio=target_ratio,
                base_price_ratio=product['current_price_ratio'],
                base_sales=product['current_sales'],
                kan5=product['kan5']
            )
            
            # Calculate metrics
            actual_price = product['base_price'] * target_ratio
            revenue = demand * actual_price
            lift = (demand / product['current_sales'] - 1) * 100
            discount_pct = (1 - target_ratio) * 100
            
            logger.info(f"    {discount_pct:.0f}% off (${actual_price:.2f}): "
                       f"{demand:.1f} units, ${revenue:.2f} revenue ({lift:+.1f}% lift)")
            
            results.append({
                'sku_id': product['sku_id'],
                'kan5': product['kan5'],
                'base_price': product['base_price'],
                'price_ratio': target_ratio,
                'actual_price': actual_price,
                'discount_percent': discount_pct,
                'predicted_demand': demand,
                'revenue': revenue,
                'lift_percent': lift
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('counterfactual_predictions.csv', index=False)
    logger.info("\nPredictions saved to counterfactual_predictions.csv")
    
    return results_df


def main():
    """Main execution function"""
    logger.info("Semi-Parametric Counterfactual Demand Model")
    logger.info("Base Price and Price Ratio Version")
    logger.info("="*60)
    
    # Train model
    model, training_data = train_model()
    
    # Demonstrate counterfactual predictions
    demonstrate_counterfactual_predictions(model, training_data)
    
    # Create individual SKU price-sales curves
    create_sku_price_sales_curves(model, training_data)
    
    # Test model persistence
    loaded_model = test_model_persistence(model)
    
    # Make predictions on new data
    predictions = predict_on_new_data(loaded_model)
    
    logger.info("\n" + "="*60)
    logger.info("Process completed successfully!")
    logger.info("Generated files:")
    logger.info("  - trained_model.pkl (trained model with all parameters)")
    logger.info("  - counterfactual_curves.png (demand/revenue curves)")
    logger.info("  - sku_price_sales_curves.png (individual SKU price-sales curves)")
    logger.info("  - counterfactual_predictions.csv (predictions on new data)")


if __name__ == "__main__":
    main()