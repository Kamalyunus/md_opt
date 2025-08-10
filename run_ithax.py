#!/usr/bin/env python3
"""
Run ITHAX algorithm with CSV input/output
"""

import json
import csv
import sys
from typing import List, Dict
from ithax_algorithm import ITHAX

def load_products_from_csv(filename: str) -> List[Dict]:
    """Load product data from CSV file"""
    products = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            products.append({
                'id': row['id'],
                'full_price': float(row['full_price']),
                'stock_units': int(row['stock_units']),
                'units_sold': int(row['units_sold'])
            })
    return products

def save_results_to_csv(ithax_instance, filename: str):
    """Save ITHAX results to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['id', 'full_price', 'stock_units', 'units_sold', 'cover', 
                      'assigned_depth', 'discount_percentage', 'discounted_price', 'stock_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for p in ithax_instance.selected_products:
            writer.writerow({
                'id': p.id,
                'full_price': p.full_price,
                'stock_units': p.stock_units,
                'units_sold': p.units_sold,
                'cover': round(p.cover, 2) if p.cover != float('inf') else 'INF',
                'assigned_depth': p.assigned_depth,
                'discount_percentage': f"{p.assigned_depth * 100:.0f}%",
                'discounted_price': round(p.full_price * (1 - p.assigned_depth), 2),
                'stock_value': round(p.full_price * p.stock_units, 2)
            })

def print_iteration_summary(iteration_history, config, total_value):
    """Print a concise iteration summary"""
    # Calculate targets
    if 'stock_value_target_percent' in config:
        target_stock_value = total_value * config['stock_value_target_percent']
    else:
        target_stock_value = config['stock_value_target']
    target_stock_depth = config['stock_depth_target']
    
    print("\nITERATION CONVERGENCE SUMMARY")
    print("=" * 80)
    print(f"Target Stock Value: ${target_stock_value:,.0f} | Target Stock Depth: {target_stock_depth*100:.1f}%")
    print("-" * 80)
    print(f"{'Iter':<4} {'Stock Value':<12} {'Stock Depth':<12} {'Products':<9} {'Val Error':<9} {'Depth Error':<10} {'Status':<8}")
    print("-" * 80)
    
    for item in iteration_history:
        val_status = "✓" if item['value_error'] < 0.05 else "✗"
        depth_status = "✓" if item['depth_error'] < 0.005 else "✗"
        overall_status = "CONV" if (val_status == "✓" and depth_status == "✓") else "ITER"
        
        print(f"{item['iteration']:<4} "
              f"${item['stock_value']:>10,.0f} "
              f"{item['stock_depth']*100:>10.2f}% "
              f"{item['n_products']:>8} "
              f"{item['value_error']*100:>7.2f}% {val_status} "
              f"{item['depth_error']*100:>7.2f}% {depth_status} "
              f"{overall_status}")

def create_matplotlib_visualization(iteration_history, config, total_value):
    """Create matplotlib visualization charts"""
    import matplotlib.pyplot as plt
    
    # Extract data
    iterations = [item['iteration'] for item in iteration_history]
    stock_values = [item['stock_value'] for item in iteration_history]
    stock_depths = [item['stock_depth'] * 100 for item in iteration_history]
    n_products = [item['n_products'] for item in iteration_history]
    value_errors = [item['value_error'] * 100 for item in iteration_history]
    depth_errors = [item['depth_error'] * 100 for item in iteration_history]
    
    # Calculate targets
    if 'stock_value_target_percent' in config:
        target_stock_value = total_value * config['stock_value_target_percent']
    else:
        target_stock_value = config['stock_value_target']
    target_stock_depth = config['stock_depth_target'] * 100
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ITHAX Algorithm Convergence Analysis', fontsize=16, fontweight='bold')
    
    # Stock Value
    ax1.plot(iterations, stock_values, 'b-o', linewidth=2, markersize=5, label='Actual Value')
    ax1.axhline(y=target_stock_value, color='r', linestyle='--', linewidth=2, label=f'Target (${target_stock_value:,.0f})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Stock Value ($)')
    ax1.set_title('Stock Value Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Stock Depth
    ax2.plot(iterations, stock_depths, 'g-o', linewidth=2, markersize=5, label='Actual Depth')
    ax2.axhline(y=target_stock_depth, color='r', linestyle='--', linewidth=2, label=f'Target ({target_stock_depth:.1f}%)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Stock Depth (%)')
    ax2.set_title('Stock Depth Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Products Count
    ax3.plot(iterations, n_products, 'm-o', linewidth=2, markersize=5, label='Selected Products')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Number of Products')
    ax3.set_title('Products Selected per Iteration')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Error Convergence
    ax4.plot(iterations, value_errors, 'b-o', linewidth=2, markersize=5, label='Stock Value Error')
    ax4.plot(iterations, depth_errors, 'g-o', linewidth=2, markersize=5, label='Stock Depth Error')
    ax4.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='Value Threshold (5%)')
    ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Depth Threshold (0.5%)')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Error (%)')
    ax4.set_title('Convergence Error (Log Scale)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_yscale('log')
    
    plt.tight_layout()
    chart_file = 'ithax_convergence.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nVisualization chart saved to: {chart_file}")

def main():
    """Main execution function"""
    # Parse arguments
    input_file = 'input_products.csv'
    output_file = 'output_results.csv'
    config_file = 'config.json'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    if len(sys.argv) > 3:
        config_file = sys.argv[3]
    
    print("ITHAX ALGORITHM EXECUTION")
    print("="*80)
    
    # Load configuration
    print("\n1. Loading configuration...")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    if 'stock_value_target_percent' in config:
        print(f"   Stock value target: {config['stock_value_target_percent']*100:.1f}% of total stock")
    else:
        print(f"   Stock value target: ${config['stock_value_target']:,.2f}")
    print(f"   Stock depth target: {config['stock_depth_target']*100:.1f}%")
    print(f"   Available depths: {config['available_depths']}")
    
    # Load product data
    print(f"\n2. Loading product data from {input_file}...")
    try:
        products_data = load_products_from_csv(input_file)
        print(f"   Loaded {len(products_data)} products")
    except FileNotFoundError:
        print(f"ERROR: File '{input_file}' not found!")
        print("\nPlease run 'python generate_data.py' first to create sample data")
        print("Or specify an existing CSV file: python run_ithax.py <input_file>")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading file: {e}")
        sys.exit(1)
    
    # Calculate total available stock value
    total_value = sum(p['full_price'] * p['stock_units'] for p in products_data)
    print(f"   Total stock value available: ${total_value:,.2f}")
    
    # Initialize and run ITHAX algorithm
    print("\n3. Running ITHAX optimization...")
    print("="*80)
    
    ithax = ITHAX(config, products_data)
    
    # Run optimization (this will print iteration details via _print_bands)
    if 'stock_value_target_percent' in config:
        target_value = total_value * config['stock_value_target_percent']
        print(f"\nTarget stock value: ${target_value:,.2f} ({config['stock_value_target_percent']*100:.1f}% of ${total_value:,.2f})")
    else:
        print(f"\nTarget stock value: ${config['stock_value_target']:,.2f}")
    print(f"Target stock depth: {config['stock_depth_target']*100:.2f}%")
    print("-"*50)
    
    result = ithax.optimize()
    
    # Print results
    print(f"\nOptimization Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Products selected: {result['n_products_selected']}")
    print(f"Final stock value: ${result['final_stock_value']:,.2f}")
    print(f"Final stock depth: {result['final_stock_depth']*100:.2f}%")
    
    # Create visualization if iteration history is available
    if 'iteration_history' in result and result['iteration_history']:
        print_iteration_summary(result['iteration_history'], config, total_value)
        create_matplotlib_visualization(result['iteration_history'], config, total_value)
    
    # Save results to CSV
    print(f"\nSaving results to {output_file}...")
    save_results_to_csv(ithax, output_file)
    print(f"   Saved {len(ithax.selected_products)} selected products")
    
    # Print discount distribution
    depth_dist = {}
    for p in ithax.selected_products:
        depth_pct = f"{p.assigned_depth*100:.0f}%"
        depth_dist[depth_pct] = depth_dist.get(depth_pct, 0) + 1
    
    print(f"\nDiscount depth distribution:")
    for depth in sorted(depth_dist.keys(), key=lambda x: float(x.rstrip('%'))):
        print(f"  {depth:>4} off: {depth_dist[depth]} products")
    
    # Show sample of selected products
    print(f"\nExample selected products (top 10 by value):")
    sorted_products = sorted(ithax.selected_products, 
                            key=lambda p: p.full_price * p.stock_units, 
                            reverse=True)[:10]
    
    print(f"{'ID':<15} {'Price':>8}  {'Stock':>5}  {'Cover':>6}  {'Depth':>6}  {'Value':>10}")
    print("-"*60)
    for p in sorted_products:
        print(f"{p.id:<15} ${p.full_price:>7.2f}  {p.stock_units:>5}  {p.cover:>6.1f}  "
              f"{p.assigned_depth*100:>5.0f}%  ${p.full_price * p.stock_units:>9.2f}")
    
    print("\nITHAX optimization complete!")
    
    # Print usage instructions
    if len(sys.argv) == 1:
        print("\nUsage:")
        print("  python run_ithax.py [input_csv] [output_csv] [config_json]")
        print("\nDefaults:")
        print(f"  Input:  {input_file}")
        print(f"  Output: {output_file}")
        print(f"  Config: {config_file}")

if __name__ == "__main__":
    main()