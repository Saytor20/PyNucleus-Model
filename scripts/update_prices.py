#!/usr/bin/env python3
"""
Price Update Script for PyNucleus

Allows manual updates to product pricing database without external API dependencies.
Offline-friendly and maintainable.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

def load_prices() -> Dict[str, Any]:
    """Load current pricing data."""
    prices_file = Path("data/product_prices.json")
    if not prices_file.exists():
        print("❌ Product prices file not found. Creating new one...")
        return create_default_prices()
    
    try:
        with open(prices_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading prices: {e}")
        return create_default_prices()

def create_default_prices() -> Dict[str, Any]:
    """Create default pricing structure."""
    return {
        "metadata": {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "currency": "USD",
            "unit": "per_ton",
            "description": "Reference pricing database for chemical products"
        },
        "prices": {
            "Methanol": 350,
            "Ammonia": 400,
            "Ethylene": 800,
            "Polyethylene": 1200,
            "Urea": 300
        },
        "market_indicators": {
            "crude_oil_brent": 85,
            "natural_gas_henry_hub": 3.5
        },
        "regional_adjustments": {
            "North_America": 1.0,
            "Europe": 1.1,
            "Asia_Pacific": 0.9
        },
        "seasonal_factors": {
            "Q1": 1.05,
            "Q2": 1.0,
            "Q3": 0.95,
            "Q4": 1.1
        }
    }

def save_prices(prices_data: Dict[str, Any]) -> bool:
    """Save pricing data to file."""
    try:
        prices_file = Path("data/product_prices.json")
        prices_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        prices_data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        with open(prices_file, 'w') as f:
            json.dump(prices_data, f, indent=2)
        
        print(f"✅ Prices saved to {prices_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving prices: {e}")
        return False

def update_product_price(prices_data: Dict[str, Any], product: str, price: float) -> bool:
    """Update a specific product price."""
    if product not in prices_data["prices"]:
        print(f"⚠️  Product '{product}' not found. Adding new product...")
    
    old_price = prices_data["prices"].get(product, 0)
    prices_data["prices"][product] = price
    
    print(f"✅ Updated {product}: ${old_price} → ${price}/ton")
    return True

def update_market_indicator(prices_data: Dict[str, Any], indicator: str, value: float) -> bool:
    """Update a market indicator."""
    if indicator not in prices_data["market_indicators"]:
        print(f"⚠️  Market indicator '{indicator}' not found. Adding new indicator...")
    
    old_value = prices_data["market_indicators"].get(indicator, 0)
    prices_data["market_indicators"][indicator] = value
    
    print(f"✅ Updated {indicator}: {old_value} → {value}")
    return True

def bulk_update_from_file(prices_data: Dict[str, Any], file_path: str) -> bool:
    """Bulk update prices from a CSV or JSON file."""
    try:
        update_file = Path(file_path)
        if not update_file.exists():
            print(f"❌ Update file not found: {file_path}")
            return False
        
        if update_file.suffix.lower() == '.json':
            with open(update_file, 'r') as f:
                updates = json.load(f)
        elif update_file.suffix.lower() == '.csv':
            import csv
            updates = {}
            with open(update_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'product' in row and 'price' in row:
                        try:
                            updates[row['product']] = float(row['price'])
                        except ValueError:
                            print(f"⚠️  Invalid price for {row['product']}: {row['price']}")
        else:
            print(f"❌ Unsupported file format: {update_file.suffix}")
            return False
        
        # Apply updates
        for product, price in updates.items():
            update_product_price(prices_data, product, price)
        
        print(f"✅ Applied {len(updates)} price updates from {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error in bulk update: {e}")
        return False

def show_current_prices(prices_data: Dict[str, Any], filter_product: Optional[str] = None):
    """Display current prices."""
    print("\n📊 Current Product Prices:")
    print("-" * 50)
    
    prices = prices_data["prices"]
    if filter_product:
        if filter_product in prices:
            print(f"{filter_product}: ${prices[filter_product]}/ton")
        else:
            print(f"❌ Product '{filter_product}' not found")
        return
    
    # Show top 20 products by price
    sorted_prices = sorted(prices.items(), key=lambda x: x[1], reverse=True)
    for product, price in sorted_prices[:20]:
        print(f"{product:<25} ${price:>8}/ton")
    
    if len(prices) > 20:
        print(f"... and {len(prices) - 20} more products")

def show_market_indicators(prices_data: Dict[str, Any]):
    """Display market indicators."""
    print("\n📈 Market Indicators:")
    print("-" * 30)
    
    indicators = prices_data["market_indicators"]
    for indicator, value in indicators.items():
        print(f"{indicator:<25} {value:>8}")

def interactive_update(prices_data: Dict[str, Any]):
    """Interactive price update mode."""
    print("\n🔄 Interactive Price Update Mode")
    print("Commands: 'product <name> <price>', 'market <indicator> <value>', 'save', 'quit'")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                break
            elif command == 'save':
                if save_prices(prices_data):
                    print("✅ Prices saved successfully")
                break
            elif command.startswith('product '):
                parts = command.split()
                if len(parts) >= 3:
                    product = ' '.join(parts[1:-1])
                    try:
                        price = float(parts[-1])
                        update_product_price(prices_data, product, price)
                    except ValueError:
                        print("❌ Invalid price value")
                else:
                    print("❌ Usage: product <name> <price>")
            elif command.startswith('market '):
                parts = command.split()
                if len(parts) >= 3:
                    indicator = ' '.join(parts[1:-1])
                    try:
                        value = float(parts[-1])
                        update_market_indicator(prices_data, indicator, value)
                    except ValueError:
                        print("❌ Invalid value")
                else:
                    print("❌ Usage: market <indicator> <value>")
            elif command == 'show':
                show_current_prices(prices_data)
            elif command == 'indicators':
                show_market_indicators(prices_data)
            else:
                print("❌ Unknown command. Use 'product', 'market', 'save', or 'quit'")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function."""
    print("🏭 PyNucleus Price Update Script")
    print("=" * 40)
    
    # Load current prices
    prices_data = load_prices()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'show':
            if len(sys.argv) > 2:
                show_current_prices(prices_data, sys.argv[2])
            else:
                show_current_prices(prices_data)
        
        elif command == 'indicators':
            show_market_indicators(prices_data)
        
        elif command == 'update' and len(sys.argv) >= 4:
            product = sys.argv[2]
            try:
                price = float(sys.argv[3])
                update_product_price(prices_data, product, price)
                save_prices(prices_data)
            except ValueError:
                print("❌ Invalid price value")
        
        elif command == 'bulk' and len(sys.argv) >= 3:
            file_path = sys.argv[2]
            if bulk_update_from_file(prices_data, file_path):
                save_prices(prices_data)
        
        elif command == 'interactive':
            interactive_update(prices_data)
        
        else:
            print("❌ Unknown command or insufficient arguments")
            print_usage()
    else:
        print_usage()

def print_usage():
    """Print usage information."""
    print("\n📋 Usage:")
    print("  python update_prices.py show [product_name]     - Show all prices or specific product")
    print("  python update_prices.py indicators              - Show market indicators")
    print("  python update_prices.py update <product> <price> - Update single product price")
    print("  python update_prices.py bulk <file.csv>         - Bulk update from CSV/JSON file")
    print("  python update_prices.py interactive             - Interactive update mode")
    print("\n📁 File formats:")
    print("  CSV: product,price")
    print("  JSON: {\"product\": price, ...}")

if __name__ == "__main__":
    main() 