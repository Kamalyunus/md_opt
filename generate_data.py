#!/usr/bin/env python3
"""
Generate sample product data for ITHAX algorithm testing
with bell-shaped distribution and long positive tail for cover metric
"""

import csv
import random
import math
import sys
from typing import List, Dict

def generate_sample_products(num_products: int = 500) -> List[Dict]:
    """
    Generate sample product data with bell-shaped cover distribution
    and long positive tail (right-skewed)
    """
    random.seed(42)  # For reproducibility
    
    products = []
    
    for i in range(num_products):
        # Generate cover values with log-normal distribution
        # This creates a bell shape with a long positive tail
        mean_log = 2.0  # Mean of log(cover) - adjusts the peak position
        std_log = 0.8   # Std of log(cover) - adjusts spread and tail length
        
        # Generate log-normal distributed cover
        cover = random.lognormvariate(mean_log, std_log)
        
        # Cap extreme values but keep long tail
        cover = min(cover, 200)  # Cap at 200 weeks
        cover = max(cover, 0.5)   # Minimum 0.5 weeks
        
        # Generate price based on product category
        # Higher cover products tend to be higher priced (slow movers)
        if cover > 50:  # Very slow movers - often luxury/specialty items
            price = random.uniform(150, 400)
        elif cover > 20:  # Slow movers
            price = random.uniform(80, 200)
        elif cover > 10:  # Normal movers
            price = random.uniform(40, 120)
        else:  # Fast movers - often everyday items
            price = random.uniform(15, 60)
        
        # Add some noise to price
        price = price * random.uniform(0.7, 1.3)
        
        # Calculate stock and sales from cover
        # Cover = stock_units / units_sold
        # Generate units_sold first based on product type
        if cover < 5:  # Fast movers
            units_sold = random.randint(30, 60)
        elif cover < 15:  # Normal movers
            units_sold = random.randint(10, 40)
        elif cover < 30:  # Slow movers
            units_sold = random.randint(3, 15)
        else:  # Very slow movers
            units_sold = max(1, random.randint(0, 5))
        
        # Calculate stock from cover and units_sold
        stock_units = int(cover * units_sold)
        
        # Add some noise to stock
        stock_units = max(10, int(stock_units * random.uniform(0.8, 1.2)))
        
        # Ensure minimum stock for high-value items
        if price > 200:
            stock_units = max(20, stock_units)
        
        products.append({
            'id': f'PROD_{i:04d}',
            'full_price': round(price, 2),
            'stock_units': stock_units,
            'units_sold': units_sold
        })
    
    return products

def save_products_to_csv(products: List[Dict], filename: str):
    """Save product data to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['id', 'full_price', 'stock_units', 'units_sold', 'cover']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for p in products:
            # Calculate cover for CSV
            cover = p['stock_units'] / p['units_sold'] if p['units_sold'] > 0 else float('inf')
            writer.writerow({
                'id': p['id'],
                'full_price': p['full_price'],
                'stock_units': p['stock_units'],
                'units_sold': p['units_sold'],
                'cover': round(cover, 2) if cover != float('inf') else 'INF'
            })

def main():
    """Main execution function"""
    # Parse arguments
    num_products = 500
    output_file = 'input_products.csv'
    
    if len(sys.argv) > 1:
        try:
            num_products = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of products: {sys.argv[1]}")
            print("Usage: python generate_data.py [num_products] [output_file]")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("SAMPLE DATA GENERATION")
    print("="*50)
    print(f"Generating {num_products} products with bell-shaped distribution...")
    print("(Log-normal distribution with long positive tail)")
    
    # Generate products
    products = generate_sample_products(num_products)
    
    # Calculate statistics
    total_value = sum(p['full_price'] * p['stock_units'] for p in products)
    avg_price = sum(p['full_price'] for p in products) / len(products)
    avg_stock = sum(p['stock_units'] for p in products) / len(products)
    
    # Save to CSV
    save_products_to_csv(products, output_file)
    
    print(f"\nSaved to: {output_file}")
    print("\nStatistics:")
    print(f"  Total stock value: ${total_value:,.2f}")
    print(f"  Average price: ${avg_price:.2f}")
    print(f"  Average stock: {avg_stock:.0f} units")
    
    # Cover distribution
    covers = []
    for p in products:
        if p['units_sold'] > 0:
            covers.append(p['stock_units'] / p['units_sold'])
    
    if covers:
        covers.sort()
        print(f"\nCover distribution (weeks to sell out):")
        print(f"  Min:     {covers[0]:>6.1f} weeks")
        print(f"  5%:      {covers[int(len(covers)*0.05)]:>6.1f} weeks")
        print(f"  25%:     {covers[int(len(covers)*0.25)]:>6.1f} weeks")
        print(f"  Median:  {covers[int(len(covers)*0.50)]:>6.1f} weeks")
        print(f"  Mean:    {sum(covers)/len(covers):>6.1f} weeks")
        print(f"  75%:     {covers[int(len(covers)*0.75)]:>6.1f} weeks")
        print(f"  95%:     {covers[int(len(covers)*0.95)]:>6.1f} weeks")
        print(f"  Max:     {covers[-1]:>6.1f} weeks")
        
        # Show distribution shape
        print("\nDistribution shape (bell-shaped with long tail):")
        
        # Create histogram bins
        bins = [0, 5, 10, 15, 20, 30, 50, 100, 200]
        bin_counts = [0] * (len(bins) - 1)
        
        for c in covers:
            for j in range(len(bins) - 1):
                if c >= bins[j] and c < bins[j+1]:
                    bin_counts[j] += 1
                    break
            else:
                if c >= bins[-1]:
                    bin_counts[-1] += 1
        
        # Display histogram
        max_count = max(bin_counts)
        for j in range(len(bin_counts)):
            if j < len(bins) - 2:
                label = f"  [{bins[j]:>3}-{bins[j+1]:>3})"
            else:
                label = f"  [{bins[j]:>3}+)  "
            
            bar_len = int(40 * bin_counts[j] / max_count) if max_count > 0 else 0
            bar = '█' * bar_len
            print(f"{label}: {bar} {bin_counts[j]} products")
    
    print("\nData generation complete!")

if __name__ == "__main__":
    main()