#!/usr/bin/env python3
"""
Legacy Nelson-Siegel Script

This script recreates the functionality from the original nelsonsiegelmodel.py
file but using the new modular architecture. It demonstrates how to migrate
from the old script-based approach to the new class-based approach.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from nelson_siegel import (
    NelsonSiegelModel,
    TreasuryNelsonSiegelModel,
    TIPSNelsonSiegelModel,
    YieldCurveAnalyzer,
    DataManager
)

# Try to import interactive components
try:
    import ipywidgets as widgets
    from IPython.display import display
    from nelson_siegel.interactive import InteractiveYieldCurveExplorer
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False
    print("⚠️ ipywidgets not available. Interactive features disabled.")


def create_sample_data():
    """Create sample yield data similar to the original script."""
    
    # Create sample dates (last 2 years of data)
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='M')
    
    # Sample Treasury data (replace TSY from original)
    tsy_maturities = [1/12, 0.25, 0.5, 1., 2., 3., 5., 10., 30.]
    np.random.seed(42)
    tsy_data = pd.DataFrame(index=dates, columns=tsy_maturities)
    
    for i, date in enumerate(dates):
        # Create realistic yield curve
        base_level = 0.03 + 0.01 * np.sin(i * 2 * np.pi / 12)  # Seasonal variation
        yields = []
        for mat in tsy_maturities:
            yield_val = base_level + 0.008 * np.log(mat + 0.1) + np.random.normal(0, 0.002)
            yields.append(max(yield_val, 0.001))  # Ensure positive yields
        tsy_data.loc[date] = yields
    
    # Sample TIPS data (replace TIPS from original)
    tips_maturities = [5., 7., 10., 20., 30.]
    tips_data = pd.DataFrame(index=dates, columns=tips_maturities)
    
    for i, date in enumerate(dates):
        # TIPS real yields (lower than nominal)
        base_real = 0.01 + 0.005 * np.sin(i * 2 * np.pi / 12)
        yields = []
        for mat in tips_maturities:
            yield_val = base_real + 0.004 * np.log(mat + 0.1) + np.random.normal(0, 0.001)
            yields.append(yield_val)  # Can be negative for TIPS
        tips_data.loc[date] = yields
    
    return {
        'TSY': tsy_data,
        'TIPS': tips_data
    }


def plot_curve_legacy(curve_data, curve_name='TSY'):
    """
    Legacy function that recreates the original plot_curve functionality
    but using the new model classes.
    """
    
    print(f"\\n📊 Analyzing {curve_name} Yield Curve (Legacy Mode)")
    print("=" * 50)
    
    # Get the latest yields (equivalent to .iloc[-1] in original)
    yields = curve_data.iloc[-1].dropna().values
    maturities = curve_data.iloc[-1].dropna().index.values
    
    print(f"Latest date: {curve_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Yields: {yields * 100}")
    print(f"Maturities: {maturities}")
    
    # Choose appropriate model based on curve type
    if curve_name.upper() == 'TSY':
        model = TreasuryNelsonSiegelModel()
    elif curve_name.upper() == 'TIPS':
        model = TIPSNelsonSiegelModel()
    else:
        # Use generic model with broad bounds
        model = NelsonSiegelModel(
            bounds=([-5, -5, -5, 0], [15, 10, 10, 10]),
            initial_guess=(0.05, 0.0, 0.0, 1.0)
        )
    
    # Fit the model
    try:
        model.fit(maturities, yields)
        factors = model.get_factors()
        
        print(f"\\nFitted Nelson-Siegel Parameters:")
        print(f"Level (β₀): {factors['Level']*100:.3f}%")
        print(f"Slope (β₁): {factors['Slope']*100:.3f}%")
        print(f"Curvature (β₂): {factors['Curvature']*100:.3f}%")
        print(f"Tau (λ): {factors['Tau']:.3f} years")
        
        # Create the plot (similar to original plot_yield_curve function)
        fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
        
        # Generate smooth curve for plotting
        smooth_maturities = np.linspace(maturities.min(), maturities.max(), 100)
        fitted_yields = model.predict(smooth_maturities)
        fitted_points = model.predict(maturities)
        
        # Top plot: Actual vs Fitted
        axs[0].plot(maturities, yields, 'bo-', label='Actual Yields', markersize=8)
        axs[0].plot(smooth_maturities, fitted_yields, 'r-', label='Fitted Yields', linewidth=2)
        axs[0].plot(maturities, fitted_points, 'r*', markersize=10, alpha=0.7)
        axs[0].set_ylabel('Yield')
        axs[0].set_title(f'Fitted Yield Curve from Nelson-Siegel Model - {curve_name}')
        axs[0].legend()
        axs[0].grid(True)
        
        # Bottom plot: Differences (like original)
        yield_difference = (yields - fitted_points) * 100  # Convert to basis points
        colors = ['red' if diff > 0 else 'green' for diff in yield_difference]
        axs[1].bar(maturities, yield_difference, color=colors, alpha=0.7)
        axs[1].set_xlabel('Maturity (years)')
        axs[1].set_ylabel('Difference (Actual - Fitted) bps')
        axs[1].set_title('Difference between Actual Yields and Fitted Yields')
        axs[1].grid(True)
        axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add RMSE information
        rmse = np.sqrt(np.mean(yield_difference**2))
        axs[1].text(0.02, 0.98, f'RMSE: {rmse:.1f} bps', 
                   transform=axs[1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Return factors for further analysis (like original)
        return model, factors
        
    except Exception as e:
        print(f"❌ Error fitting model: {e}")
        return None, None


def historic_params_legacy(yields_data, curve_name='TSY'):
    """
    Legacy function that recreates the original historic_params functionality.
    """
    
    print(f"\\n📈 Calculating Historical Parameters for {curve_name}")
    print("=" * 50)
    
    curve_data = yields_data[curve_name]
    
    # Choose appropriate model
    if curve_name.upper() == 'TSY':
        model_class = TreasuryNelsonSiegelModel
    elif curve_name.upper() == 'TIPS':
        model_class = TIPSNelsonSiegelModel
    else:
        model_class = NelsonSiegelModel
    
    factors_list = []
    dates_list = []
    failed_count = 0
    
    print(f"Processing {len(curve_data)} dates...")
    
    for i, (date, row) in enumerate(curve_data.iterrows()):
        if i % 6 == 0:  # Progress every 6 months
            print(f"Progress: {i}/{len(curve_data)} ({100*i/len(curve_data):.1f}%)")
        
        # Get clean data
        clean_data = row.dropna()
        if len(clean_data) < 4:  # Need minimum 4 points
            failed_count += 1
            continue
        
        try:
            model = model_class()
            model.fit(clean_data.index.values, clean_data.values)
            factors = model.get_factors()
            
            factors_list.append([
                factors['Level'],
                factors['Slope'],
                factors['Curvature'],
                factors['Tau']
            ])
            dates_list.append(date)
            
        except Exception:
            failed_count += 1
            continue
    
    if not factors_list:
        print("❌ No successful fits found")
        return pd.DataFrame()
    
    # Create factors DataFrame (like original)
    factors_df = pd.DataFrame(
        factors_list,
        columns=['Level', 'Slope', 'Curvature', 'Tau'],
        index=dates_list
    )
    
    print(f"✅ Successfully fitted {len(factors_df)} out of {len(curve_data)} dates")
    print(f"❌ Failed fits: {failed_count}")
    
    return factors_df


def plot_historical_factors_legacy(factors_df, curve_name):
    """Plot historical factors like the original script."""
    
    if factors_df.empty:
        print(f"No factors to plot for {curve_name}")
        return
    
    print(f"\\n📊 Plotting Historical Factors for {curve_name}")
    
    for factor in factors_df.columns:
        plt.figure(figsize=(12, 6))
        
        data = factors_df[factor]
        if factor != 'Tau':
            data = data * 100  # Convert to percentage
        
        plt.plot(factors_df.index, data, linewidth=2, alpha=0.8)
        
        unit = 'years' if factor == 'Tau' else '%'
        plt.title(f'{curve_name} - {factor} Factor Evolution', fontsize=14, fontweight='bold')
        plt.ylabel(f'{factor} ({unit})')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add statistics
        mean_val = data.mean()
        std_val = data.std()
        plt.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()


def create_interactive_widget_legacy(curve_data, curve_name='TSY'):
    """
    Recreate the original ipywidgets functionality using the new interactive module.
    """
    
    if not HAS_WIDGETS:
        print("❌ Interactive widgets not available. Install ipywidgets to use this feature.")
        return
    
    print(f"🎛️ Creating Interactive Widget for {curve_name}")
    
    # Map curve names to new system
    bond_type = 'treasury' if curve_name.upper() == 'TSY' else 'tips'
    
    try:
        explorer = InteractiveYieldCurveExplorer(bond_type)
        
        # Load current data from our sample
        latest_yields = curve_data.iloc[-1].dropna()
        explorer.current_yields = latest_yields.values
        explorer.current_maturities = latest_yields.index.values
        
        # Create and display the dashboard
        dashboard = explorer.create_full_interactive_dashboard()
        display(dashboard)
        
    except Exception as e:
        print(f"❌ Error creating interactive widget: {e}")


def main():
    """
    Main function that recreates the original script workflow
    but using the new modular architecture.
    """
    
    print("🏛️ NELSON-SIEGEL LEGACY ANALYSIS")
    print("=" * 50)
    print("This script recreates the original functionality using the new modular system.")
    print()
    
    # Create sample data (equivalent to reading from Excel in original)
    print("📊 Creating sample yield data...")
    yields_data = create_sample_data()
    
    # Plot current curves (equivalent to plot_curve() calls in original)
    print("\\n🎯 CURRENT YIELD CURVE ANALYSIS")
    print("=" * 40)
    
    tsy_model, tsy_factors = plot_curve_legacy(yields_data['TSY'], 'TSY')
    tips_model, tips_factors = plot_curve_legacy(yields_data['TIPS'], 'TIPS')
    
    # Historical analysis (equivalent to historic_params() calls in original)
    print("\\n📈 HISTORICAL FACTOR ANALYSIS")
    print("=" * 35)
    
    factors_tsy = historic_params_legacy(yields_data, 'TSY')
    factors_tips = historic_params_legacy(yields_data, 'TIPS')
    
    # Plot historical factors (equivalent to plotting loops in original)
    if not factors_tsy.empty:
        plot_historical_factors_legacy(factors_tsy, 'TSY')
    
    if not factors_tips.empty:
        plot_historical_factors_legacy(factors_tips, 'TIPS')
    
    # Interactive widgets (equivalent to widgets.interactive in original)
    if HAS_WIDGETS:
        print("\\n🎛️ INTERACTIVE EXPLORATION")
        print("=" * 30)
        print("Creating interactive widgets...")
        
        create_interactive_widget_legacy(yields_data['TSY'], 'TSY')
        create_interactive_widget_legacy(yields_data['TIPS'], 'TIPS')
    
    print("\\n✅ Legacy analysis completed!")
    print("💡 For modern usage, see examples/basic_usage.py or the Jupyter notebook.")
    
    return {
        'yields_data': yields_data,
        'tsy_factors': factors_tsy,
        'tips_factors': factors_tips,
        'current_models': {'TSY': tsy_model, 'TIPS': tips_model}
    }


if __name__ == "__main__":
    results = main()
