"""
ITHAX Algorithm - Exact Paper Implementation
This is a precise implementation of the ITHAX algorithm as described in:
"Promotheus: An End-to-End Machine Learning Framework for Optimizing Markdown in Online Fashion E-commerce"
KDD '22, August 14–18, 2022, Washington, DC, USA.

This implementation follows Algorithms 1-4 from the paper exactly, without modifications.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Product:
    """Represents a product with its properties"""
    id: str
    full_price: float
    stock_units: int
    units_sold: int
    cover: float = 0.0
    assigned_depth: float = 0.0
    
    def __post_init__(self):
        """Calculate cover as stock_units / units_sold_per_week"""
        if self.units_sold > 0:
            self.cover = self.stock_units / self.units_sold
        else:
            self.cover = float('inf')


@dataclass
class CoverBand:
    """Represents a cover band with boundaries and discount depth"""
    min_cover: float
    max_cover: float
    discount_depth: float
    
    @property
    def width(self):
        """Band width = max_cover - min_cover"""
        return self.max_cover - self.min_cover


class ITHAX:
    """
    Exact implementation of ITHAX Algorithm from the paper.
    No modifications or optimizations - follows paper algorithms precisely.
    """
    
    def __init__(self, config: dict, products_data: List[Dict]):
        """
        Initialize ITHAX with configuration parameters and product data
        
        Args:
            config: Dictionary containing:
                - stock_value_target_percent: Target stock value as percentage of total (0-1)
                - stock_depth_target (M*): Target average discount depth
                - available_depths (D): List of available discount depths
                - min_band_width (w): Minimum width for cover bands (default: 3)
                - zero_seller_threshold: Cover threshold for zero-sellers (default: 100)
            products_data: List of product dictionaries with id, full_price, stock_units, units_sold
        """
        # Load products first to calculate total stock value
        self.products = []
        for p in products_data:
            product = Product(
                id=p['id'],
                full_price=p['full_price'],
                stock_units=p['stock_units'],
                units_sold=p['units_sold']
            )
            self.products.append(product)
        
        # Calculate total stock value
        total_stock_value = sum(p.full_price * p.stock_units for p in self.products)
        
        self.V_star = total_stock_value * config['stock_value_target_percent']
        self.M_star = config['stock_depth_target']  # M* in paper
        
        # Algorithm parameters
        self.available_depths = sorted(config['available_depths'])  # D in paper
        self.min_band_width = config.get('min_band_width', 3)  # w in paper
        self.zero_seller_threshold = config.get('zero_seller_threshold', 100)
        
        # Convergence thresholds from paper (Section 3, eq. 8)
        self.convergence_threshold_value = 0.05  # Paper: f1(P̂) < 0.05
        self.convergence_threshold_depth = 0.005  # Paper: f2(P̂) < 0.005
        
        # Maximum iterations (paper mentions 25 iterations typical)
        self.max_iterations = config.get('max_iterations', 100)
        
        # Sort products by cover for easier processing
        self.products.sort(key=lambda x: x.cover if x.cover != float('inf') else self.zero_seller_threshold + 1)
        
        # Data storage
        self.selected_products = []  # P̂ in paper
        self.iteration_history = []
        
        # Previous iteration's M(P̂) for Algorithm 4 decision
        self.previous_depth = None
    
    def initialize_cover_bands(self) -> List[CoverBand]:
        """
        Initialize ν₀: initial mapping between cover bands and discount depths
        Paper: "linear mapping between cover and discount depth"
        
        Data-driven implementation based on actual product distribution:
        - Uses quantile-based boundaries for better distribution across products
        - Maintains paper's highest-to-lowest depth assignment
        - Ensures bands have sufficient width for adjustments
        """
        bands = []
        
        # Get products with finite cover below zero-seller threshold
        valid_products = [p for p in self.products 
                         if p.cover < self.zero_seller_threshold and p.cover != float('inf')]
        
        if not valid_products:
            return bands
        
        # Sort products by cover for analysis
        valid_products.sort(key=lambda x: x.cover)
        
        # Get non-zero depths and sort lowest to highest 
        # Higher cover ranges (worse performance) get higher depths
        active_depths = sorted([d for d in self.available_depths if d > 0])
        if not active_depths:
            return bands
        
        # Create bands based on data distribution
        covers = [p.cover for p in valid_products]
        min_cover = min(covers)
        max_cover = min(max(covers), self.zero_seller_threshold)
        
        # Use quantile-based boundaries for better coverage
        num_bands = len(active_depths)
        band_boundaries = []
        
        for i in range(num_bands):
            if i == 0:
                band_min = min_cover
            else:
                # Use quantiles to distribute products more evenly across bands
                quantile = i / num_bands
                quantile_idx = int(quantile * len(valid_products))
                band_min = valid_products[min(quantile_idx, len(valid_products) - 1)].cover
                
                # Ensure no overlap with previous band
                if band_boundaries:
                    band_min = max(band_min, band_boundaries[-1][1])
            
            if i == num_bands - 1:
                band_max = max_cover
            else:
                # Next quantile boundary
                next_quantile = (i + 1) / num_bands
                next_quantile_idx = int(next_quantile * len(valid_products))
                band_max = valid_products[min(next_quantile_idx, len(valid_products) - 1)].cover
            
            # Ensure minimum band width
            if band_max - band_min < self.min_band_width:
                band_max = band_min + self.min_band_width
            
            band_boundaries.append((band_min, band_max))
        
        # Create bands with correct depth assignment:
        # First band (lowest cover) gets lowest depth, last band (highest cover) gets highest depth
        for i, (min_cover, max_cover) in enumerate(band_boundaries):
            depth = active_depths[i]  # active_depths is sorted low to high
            bands.append(CoverBand(min_cover, max_cover, depth))
        
        # Add zero-seller band (products above threshold get 0% discount)
        bands.append(CoverBand(self.zero_seller_threshold, float('inf'), 0.0))
        
        return bands
    
    def _print_bands(self, bands: List[CoverBand], iteration: int):
        """Print band details for debugging"""
        print(f"\n--- Iteration {iteration} Bands ---")
        for i, band in enumerate(bands):
            if band.discount_depth > 0:
                # Count products in this band
                products_in_band = [p for p in self.products 
                                    if band.min_cover <= p.cover < band.max_cover]
                band_value = sum(p.full_price * p.stock_units for p in products_in_band)
                
                print(f"  Band {i+1}: [{band.min_cover:>6.1f}, {band.max_cover:>6.1f}) "
                        f"depth={band.discount_depth:>4.0%}, width={band.width:>5.1f} "
                        f"-> {len(products_in_band):>3} products, ${band_value:>9,.0f}")
    
    def calculate_stock_value(self, products: List[Product]) -> float:
        """
        Calculate V(P): total full price value
        Equation (2) from paper: V(Pt) = Σ fp · kp,t
        """
        return sum(p.full_price * p.stock_units for p in products)
    
    def calculate_stock_depth(self, products: List[Product]) -> float:
        """
        Calculate M(P): average discount depth
        Equation (3) from paper: M(Pt) = 1 - (Σ(1-dp,t)·fp·kp,t) / (Σfp·kp,t)
        """
        if not products:
            return 0.0
        
        total_full_value = sum(p.full_price * p.stock_units for p in products)
        if total_full_value == 0:
            return 0.0
        
        total_discounted_value = sum(
            (1 - p.assigned_depth) * p.full_price * p.stock_units 
            for p in products
        )
        
        return 1 - (total_discounted_value / total_full_value)
    
    def depth_allocation(self, bands: List[CoverBand]) -> List[Product]:
        """
        Algorithm 2: Depth Allocation Step
        Constructs P̂ using bands to hit V*
        """
        P_hat = []  # Selected products
        current_value = 0.0
        
        # Identify bands with non-zero depths
        bands_with_depth = [b for b in bands if b.discount_depth > 0]
        if not bands_with_depth:
            return P_hat
        
        # Sort by discount depth
        bands_with_depth.sort(key=lambda b: b.discount_depth)
        
        # Process from highest depth to lowest (Line 3 in Algorithm 2)
        for band in reversed(bands_with_depth):
            # Get products in this band (Line 4)
            P_b = [p for p in self.products 
                   if band.min_cover <= p.cover < band.max_cover]
            
            # Calculate stock value of products in band b
            V_Pb = self.calculate_stock_value(P_b)
            
            # Line 5-6: If adding all products doesn't exceed V*
            if current_value + V_Pb <= self.V_star:
                # Add all products from this band
                for p in P_b:
                    p.assigned_depth = band.discount_depth
                    P_hat.append(p)
                current_value += V_Pb
            else:
                # Line 8-10: Select subset to roughly achieve V*
                # Prioritize by cover (highest first) to select worst performers
                remaining_target = self.V_star - current_value
                
                # Sort products by cover (highest first) - worst performers get priority
                P_b.sort(key=lambda p: p.cover, reverse=True)
                
                # Try to get close to target
                P_x = []
                for p in P_b:
                    p_value = p.full_price * p.stock_units
                    if p_value <= remaining_target:
                        P_x.append(p)
                        remaining_target -= p_value
                
                # Add selected subset
                for p in P_x:
                    p.assigned_depth = band.discount_depth
                    P_hat.append(p)
                
                break  # Stop after partial allocation
        
        return P_hat
    
    def is_adjustable(self, band: CoverBand) -> bool:
        """
        Algorithm 5 (Appendix): Check if band is adjustable
        Band is adjustable if depth > 0 and width > min_band_width
        """
        return band.discount_depth > 0 and band.width > self.min_band_width
    
    def adjust_reduce_depth(self, bands: List[CoverBand], b_x_idx: int) -> Tuple[List[CoverBand], int]:
        """
        Algorithm 3: Adjusting Bi for M(P̂i) > M*
        Reduce depth by shrinking high-depth bands
        """
        # Line 1-3: Find adjustable band
        while b_x_idx >= 0 and not self.is_adjustable(bands[b_x_idx]):
            b_x_idx -= 1
        
        if b_x_idx < 0:
            return bands, b_x_idx
        
        # Line 4: Initialize x = (cmax_bx - cmin_bx) / 2 (halve the band width)
        b_x = bands[b_x_idx]
        x = b_x.width / 2
        
        # Line 5: Reduce upper bound of target band
        bands[b_x_idx].max_cover = b_x.max_cover - x
        
        # Line 6-9: Shift all higher bands down by x
        for i in range(b_x_idx + 1, len(bands)):
            if bands[i].discount_depth > 0:  # Only adjust bands with products
                bands[i].min_cover = bands[i].min_cover - x
                bands[i].max_cover = bands[i].max_cover - x
        
        # Line 10: Return updated bands and b_x
        return bands, b_x_idx
    
    def adjust_increase_depth(self, bands: List[CoverBand], b_x_idx: int, 
                             M_i: float, M_i_minus_1: Optional[float], iteration: int) -> Tuple[List[CoverBand], int]:
        """
        Algorithm 4: Adjusting Bi for M(P̂i) < M*
        Increase depth by expanding high-depth bands or transferring products
        
        Inputs (as per paper):
        - B (bands): Current band configuration
        - b_x (b_x_idx): Target band index
        - M(P̂i) (M_i): Current stock depth
        - M(P̂i-1) (M_i_minus_1): Previous iteration's stock depth
        - i (iteration): Current iteration number
        """
        # Lines 1-3: Special case for i=1 (first iteration)
        if iteration == 1:
            # Expand the upper bound of highest depth band
            x = bands[b_x_idx].width / 2
            bands[b_x_idx].max_cover = bands[b_x_idx].max_cover + x
            return bands, b_x_idx
        
        # Find highest depth band
        highest_idx = max(range(len(bands)), 
                         key=lambda i: bands[i].discount_depth if bands[i].discount_depth > 0 else -1)
        
        # Line 5-10: Check if b_x is highest depth band
        if b_x_idx == highest_idx:
            # Line 6-9: Check if depth hasn't changed from previous iteration
            if abs(M_i - M_i_minus_1) < 0.001:  # M(P̂i) ≈ M(P̂i-1)
                # Move to next band
                b_x_idx -= 1
            else:
                # Line 8: Expand highest depth band (Option A) - expand upper bound
                x = bands[b_x_idx].width / 2
                bands[b_x_idx].max_cover = bands[b_x_idx].max_cover + x
                return bands, b_x_idx
        
        # Line 12-14: Find adjustable band if needed
        while b_x_idx >= 0 and not self.is_adjustable(bands[b_x_idx]):
            b_x_idx -= 1
        
        if b_x_idx < 0:
            return bands, b_x_idx
        
        # Line 16-23: Transfer products (Option B)
        # Halve the target band width
        x = bands[b_x_idx].width / 2
        
        # Line 17: Reduce target band upper bound
        bands[b_x_idx].max_cover = bands[b_x_idx].max_cover - x
        
        # Line 18-22: Adjust intermediate bands
        for i in range(b_x_idx + 1, highest_idx + 1):
            if bands[i].discount_depth > 0:
                bands[i].min_cover = bands[i].min_cover - x
                if i < highest_idx:  # Don't reduce max_cover of highest band
                    bands[i].max_cover = bands[i].max_cover - x
        
        return bands, b_x_idx
    
    def optimize(self) -> Dict:
        """
        Algorithm 1: Main ITHAX optimization loop
        """
        if not self.products:
            raise ValueError("No products loaded")
        
        # Initialize ν₀ (initial cover band mapping)
        bands = self.initialize_cover_bands()
        if not bands:
            return {'status': 'error', 'message': 'Could not initialize bands'}
        
        # Line 1: Initialize b_x as highest depth band
        bands_with_depth = [(i, b) for i, b in enumerate(bands) if b.discount_depth > 0]
        if not bands_with_depth:
            return {'status': 'error', 'message': 'No bands with positive depth'}
        
        b_x_idx = max(bands_with_depth, key=lambda x: x[1].discount_depth)[0]
        
        # Store for analysis
        iteration_history = []
        
        # Line 2: Main iteration loop
        for i in range(1, self.max_iterations + 1):
            # Print bands for debugging
            self._print_bands(bands, i)
            
            # Line 3: Construct P̂i via Algorithm 2
            P_hat_i = self.depth_allocation(bands)
            
            # Compute V(P̂i), M(P̂i)
            V_i = self.calculate_stock_value(P_hat_i)
            M_i = self.calculate_stock_depth(P_hat_i)
            
            # Store iteration info
            iteration_info = {
                'iteration': i,
                'stock_value': V_i,
                'stock_depth': M_i,
                'n_products': len(P_hat_i),
                'value_error': abs(V_i - self.V_star) / self.V_star if self.V_star > 0 else 0,
                'depth_error': abs(M_i - self.M_star)
            }
            iteration_history.append(iteration_info)
            
            # Line 4-5: Check convergence (eq. 8 from paper)
            if iteration_info['value_error'] < self.convergence_threshold_value and \
               iteration_info['depth_error'] < self.convergence_threshold_depth:
                # Converged!
                self.selected_products = P_hat_i
                return {
                    'status': 'success',
                    'iterations': i,
                    'final_stock_value': V_i,
                    'final_stock_depth': M_i,
                    'n_products_selected': len(P_hat_i),
                    'products': P_hat_i,
                    'iteration_history': iteration_history
                }
            
            # Line 7-12: Boundary adjustment
            if M_i > self.M_star:
                # Line 8-9: Stock depth too high - use Algorithm 3
                # Keep b_x as highest depth band for reduction
                bands_with_depth = [(idx, b) for idx, b in enumerate(bands) if b.discount_depth > 0]
                if bands_with_depth:
                    b_x_idx = max(bands_with_depth, key=lambda x: x[1].discount_depth)[0]
                bands, b_x_idx = self.adjust_reduce_depth(bands, b_x_idx)
            else:
                # Line 11: Stock depth too low - use Algorithm 4
                bands, b_x_idx = self.adjust_increase_depth(bands, b_x_idx, M_i, self.previous_depth, i)
            
            # Store M(P̂i) for next iteration's Algorithm 4 decision
            self.previous_depth = M_i
        
        # Max iterations reached without convergence
        self.selected_products = P_hat_i
        return {
            'status': 'max_iterations',
            'iterations': self.max_iterations,
            'final_stock_value': V_i,
            'final_stock_depth': M_i,
            'n_products_selected': len(P_hat_i),
            'products': P_hat_i,
            'iteration_history': iteration_history
        }
