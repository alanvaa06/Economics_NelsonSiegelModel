#!/usr/bin/env python3
"""
Basic usage examples for the Nelson-Siegel yield curve model.

This script demonstrates the fundamental functionality of the library
including single curve fitting, parameter estimation, and basic visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from nelson_siegel import (
    NelsonSiegelModel,
    TreasuryNelsonSiegelModel, 
    TIPSNelsonSiegelModel,
    YieldCurveAnalyzer,
    YieldCurvePlotter
)


def example_1_basic_model_fitting():
    """Example 1: Basic Nelson-Siegel model fitting."""
    print("=" * 60)
    print("Example 1: Basic Nelson-Siegel Model Fitting")
    print("=" * 60)
    
    # Sample yield curve data (maturity in years, yield as decimal)
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    yields = np.array([0.054, 0.053, 0.051, 0.048, 0.046, 0.045, 0.044, 0.043, 0.046, 0.045])
    
    print(f"Input data:")
    print(f"Maturities: {maturities}")
    print(f"Yields: {(yields * 100).round(1)} %")
    
    # Create and fit model
    model = NelsonSiegelModel()
    model.fit(maturities, yields)
    
    # Get fitted parameters
    factors = model.get_factors()
    print(f"\nFitted Nelson-Siegel Parameters:")
    print(f"Level (β₀): {factors['Level'] * 100:.3f}%")
    print(f"Slope (β₁): {factors['Slope'] * 100:.3f}%")
    print(f"Curvature (β₂): {factors['Curvature'] * 100:.3f}%")
    print(f"Tau (λ): {factors['Tau']:.3f} years")
    
    # Calculate fitted yields and deviations
    fitted_yields = model.predict(maturities)
    deviations = model.calculate_deviations(maturities, yields)
    rmse = np.sqrt(np.mean(deviations**2)) * 10000  # Convert to basis points
    
    print(f"\nFit Quality:")
    print(f"RMSE: {rmse:.1f} basis points")
    print(f"Max deviation: {np.max(np.abs(deviations)) * 10000:.1f} basis points")
    
    # Bond classification
    classifications = model.classify_bonds(maturities, yields)
    print(f"\nBond Classifications:")
    for mat, cls, dev in zip(maturities, classifications, deviations):
        print(f"{mat:4.1f}Y: {cls:9s} ({dev*10000:+5.1f} bps)")


def example_2_treasury_vs_tips():
    """Example 2: Compare Treasury and TIPS models."""
    print("\n" + "=" * 60)
    print("Example 2: Treasury vs TIPS Model Comparison")
    print("=" * 60)
    
    # Treasury yield data (nominal yields)
    tsy_maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])
    tsy_yields = np.array([0.051, 0.048, 0.046, 0.045, 0.044, 0.043, 0.046, 0.045])
    
    # TIPS yield data (real yields, typically lower)
    tips_maturities = np.array([5, 7, 10, 20, 30])
    tips_yields = np.array([0.010, 0.012, 0.015, 0.018, 0.019])
    
    # Fit Treasury model
    tsy_model = TreasuryNelsonSiegelModel()
    tsy_model.fit(tsy_maturities, tsy_yields)
    tsy_factors = tsy_model.get_factors()
    
    # Fit TIPS model
    tips_model = TIPSNelsonSiegelModel()
    tips_model.fit(tips_maturities, tips_yields)
    tips_factors = tips_model.get_factors()
    
    print("Treasury (Nominal) Factors:")
    for factor, value in tsy_factors.items():
        unit = "%" if factor != "Tau" else "years"
        multiplier = 100 if factor != "Tau" else 1
        print(f"  {factor:10s}: {value * multiplier:6.3f} {unit}")
    
    print("\nTIPS (Real) Factors:")
    for factor, value in tips_factors.items():
        unit = "%" if factor != "Tau" else "years"
        multiplier = 100 if factor != "Tau" else 1
        print(f"  {factor:10s}: {value * multiplier:6.3f} {unit}")
    
    # Calculate implied inflation expectations (simplified)
    # Note: This is a simplified calculation for illustration
    common_maturities = np.array([5, 7, 10, 20, 30])
    tsy_common = tsy_model.predict(common_maturities)
    tips_common = tips_model.predict(common_maturities)
    implied_inflation = tsy_common - tips_common
    
    print(f"\nImplied Inflation Expectations:")
    for mat, inf in zip(common_maturities, implied_inflation):
        print(f"  {mat:2.0f}Y: {inf * 100:.2f}%")


def example_3_yield_curve_analyzer():
    """Example 3: Using the high-level YieldCurveAnalyzer."""
    print("\n" + "=" * 60)
    print("Example 3: High-Level Yield Curve Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = YieldCurveAnalyzer()
    
    # Manual yield data for analysis
    yields_data = {
        1.0: 0.025,   # 1Y: 2.5%
        2.0: 0.028,   # 2Y: 2.8%
        3.0: 0.030,   # 3Y: 3.0%
        5.0: 0.032,   # 5Y: 3.2%
        10.0: 0.035,  # 10Y: 3.5%
        30.0: 0.038   # 30Y: 3.8%
    }
    
    # Analyze single curve
    result = analyzer.analyze_single_curve('treasury', yields_data=yields_data)
    
    print("Analysis Results:")
    print(f"Bond Type: {result['bond_type']}")
    print(f"RMSE: {result['rmse'] * 10000:.1f} basis points")
    
    print(f"\nNelson-Siegel Factors:")
    for factor, value in result['factors'].items():
        unit = "%" if factor != "Tau" else "years"
        multiplier = 100 if factor != "Tau" else 1
        print(f"  {factor:10s}: {value * multiplier:6.3f} {unit}")
    
    print(f"\nBond Classifications:")
    mats = result['maturities']
    devs = result['deviations']
    classes = result['bond_classification']
    
    for mat, dev, cls in zip(mats, devs, classes):
        print(f"  {mat:4.1f}Y: {cls:9s} ({dev*10000:+5.1f} bps)")


def example_4_visualization():
    """Example 4: Visualization with YieldCurvePlotter."""
    print("\n" + "=" * 60)
    print("Example 4: Yield Curve Visualization")
    print("=" * 60)
    
    # Create sample data
    analyzer = YieldCurveAnalyzer()
    yields_data = {
        0.25: 0.054,
        1.0: 0.051,
        2.0: 0.048,
        5.0: 0.045,
        10.0: 0.043,
        30.0: 0.045
    }
    
    # Analyze curve
    result = analyzer.analyze_single_curve('treasury', yields_data=yields_data)
    
    # Create plotter and generate visualization
    plotter = YieldCurvePlotter()
    
    print("Generating yield curve fit plot...")
    fig = plotter.plot_yield_curve_fit(
        result, 
        title="Example Treasury Yield Curve Analysis",
        show_deviations=True
    )
    
    # Save the plot
    plt.savefig('example_yield_curve_fit.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'example_yield_curve_fit.png'")
    
    # Show the plot (comment out if running in headless environment)
    plt.show()


def example_5_custom_bounds():
    """Example 5: Using custom optimization bounds."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Optimization Bounds")
    print("=" * 60)
    
    # Sample data with potentially unusual characteristics
    maturities = np.array([1, 2, 5, 10, 30])
    yields = np.array([0.06, 0.055, 0.045, 0.04, 0.042])  # Inverted curve
    
    # Default model
    default_model = NelsonSiegelModel()
    default_model.fit(maturities, yields)
    default_factors = default_model.get_factors()
    
    # Custom model with tighter bounds
    custom_bounds = (
        [0.03, -0.05, -0.02, 0.5],  # Lower bounds
        [0.08, 0.02, 0.02, 5.0]     # Upper bounds
    )
    custom_initial = (0.05, -0.02, 0.0, 2.0)
    
    custom_model = NelsonSiegelModel(
        bounds=custom_bounds, 
        initial_guess=custom_initial
    )
    custom_model.fit(maturities, yields)
    custom_factors = custom_model.get_factors()
    
    print("Default Model Factors:")
    for factor, value in default_factors.items():
        unit = "%" if factor != "Tau" else "years"
        multiplier = 100 if factor != "Tau" else 1
        print(f"  {factor:10s}: {value * multiplier:6.3f} {unit}")
    
    print("\nCustom Bounds Model Factors:")
    for factor, value in custom_factors.items():
        unit = "%" if factor != "Tau" else "years"
        multiplier = 100 if factor != "Tau" else 1
        print(f"  {factor:10s}: {value * multiplier:6.3f} {unit}")
    
    # Compare fit quality
    default_rmse = np.sqrt(np.mean(default_model.calculate_deviations(maturities, yields)**2)) * 10000
    custom_rmse = np.sqrt(np.mean(custom_model.calculate_deviations(maturities, yields)**2)) * 10000
    
    print(f"\nFit Quality Comparison:")
    print(f"Default Model RMSE: {default_rmse:.1f} basis points")
    print(f"Custom Model RMSE:  {custom_rmse:.1f} basis points")


def main():
    """Run all examples."""
    print("Nelson-Siegel Yield Curve Model - Basic Usage Examples")
    print("=" * 60)
    
    try:
        example_1_basic_model_fitting()
        example_2_treasury_vs_tips()
        example_3_yield_curve_analyzer()
        example_4_visualization()
        example_5_custom_bounds()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    main()
