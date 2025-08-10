# ITHAX Algorithm Implementation

A Python implementation of the ITHAX (Iterative Target-based Heuristic Algorithm for eXact markdown optimization) algorithm for optimizing product discounts in e-commerce.

## Overview

This project implements the ITHAX algorithm as described in the paper "Promotheus: An End-to-End Machine Learning Framework for Optimizing Markdown in Online Fashion E-commerce" (KDD '22). The algorithm optimally selects products for discount campaigns to achieve specific stock value and discount depth targets.

## Features

- **Exact Algorithm Implementation**: Follows the paper's algorithms 1-4 precisely
- **Data Generation**: Creates realistic sample product data with bell-shaped cover distribution
- **Visualization**: Generates convergence charts and iteration summaries
- **CSV Input/Output**: Easy integration with existing data pipelines
- **Configurable Parameters**: Flexible configuration through JSON files

## Files

- `ithax_algorithm.py` - Core ITHAX algorithm implementation
- `run_ithax.py` - Main execution script with CSV I/O
- `generate_data.py` - Sample product data generator
- `config.json` - Algorithm configuration parameters
- `input_products.csv` - Input product data (generated)
- `output_results.csv` - Results with selected products and discounts
- `ithax_convergence.png` - Visualization of algorithm convergence

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv md_opt
source md_opt/bin/activate  # On Windows: md_opt\Scripts\activate
pip install matplotlib pandas
```

## Quick Start

1. **Generate sample data**:
   ```bash
   python generate_data.py
   ```

2. **Run ITHAX optimization**:
   ```bash
   python run_ithax.py
   ```

3. **View results**:
   - Check `output_results.csv` for selected products
   - View `ithax_convergence.png` for convergence visualization

## Configuration

Edit `config.json` to customize algorithm parameters:

```json
{
  "stock_value_target_percent": 0.40,  // Target 40% of total stock value
  "stock_depth_target": 0.30,          // Target 30% average discount depth
  "available_depths": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  // Discount levels
  "min_band_width": 12,                // Minimum band width for adjustments
  "max_iterations": 25,                // Maximum optimization iterations
  "zero_seller_threshold": 100         // Cover threshold for zero-sellers
}
```

## Input Data Format

The algorithm expects CSV input with these columns:

| Column | Description |
|--------|-------------|
| `id` | Product identifier |
| `full_price` | Original product price |
| `stock_units` | Available inventory |
| `units_sold` | Weekly sales rate |

The algorithm calculates "cover" = `stock_units / units_sold` (weeks to sell out).

## Algorithm Details

The ITHAX algorithm works through iterative band adjustments:

1. **Initialize** cover bands with discount depths
2. **Allocate** products to bands to hit stock value target
3. **Check convergence** against value and depth targets
4. **Adjust bands** based on whether current depth is above/below target
5. **Repeat** until convergence or max iterations

### Key Metrics

- **Stock Value**: Total value of selected products at full price
- **Stock Depth**: Value-weighted average discount percentage
- **Cover**: Inventory weeks (stock_units ÷ units_sold)

## Usage Examples

### Basic Usage
```bash
python run_ithax.py
```

### Custom Files
```bash
python run_ithax.py my_products.csv my_results.csv my_config.json
```

### Generate Custom Data
```bash
python generate_data.py 1000 my_products.csv
```

## Output

The algorithm produces:

1. **Console output** with iteration details and convergence summary
2. **CSV results** with selected products and assigned discounts
3. **Visualization charts** showing convergence behavior

### Sample Output
```
Target Stock Value: $523,891 | Target Stock Depth: 30.0%
Iter Stock Value Stock Depth Products Val Error Depth Error Status
   1  $524,156     29.82%      205    0.05% ✓    0.18% ✓    CONV
```

## Algorithm Performance

- Typically converges in 5-15 iterations
- Handles 100-10,000+ products efficiently  
- Convergence thresholds: <5% value error, <0.5% depth error

## References

Based on the paper:
```
Promotheus: An End-to-End Machine Learning Framework for Optimizing Markdown in Online Fashion E-commerce
KDD '22, August 14–18, 2022, Washington, DC, USA.
```

## License

This implementation is for research and educational purposes.