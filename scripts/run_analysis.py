#!/usr/bin/env python3
"""
Standalone script for running Nelson-Siegel yield curve analysis.

This script provides a command-line interface for performing yield curve
analysis using the nelson_siegel library.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nelson_siegel import YieldCurveAnalyzer, YieldCurvePlotter


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nelson-Siegel Yield Curve Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Treasury curves for the last year
  python run_analysis.py treasury --days 365
  
  # Compare Treasury and TIPS with specific dates
  python run_analysis.py both --start 2020-01-01 --end 2023-12-31
  
  # Analyze TIPS with custom output directory
  python run_analysis.py tips --output ./results --save-plots
        """
    )
    
    parser.add_argument(
        'bond_type',
        choices=['treasury', 'tips', 'both'],
        help='Type of bonds to analyze'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--end', 
        type=str,
        help='End date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        help='Number of days to look back from today (alternative to --start)'
    )
    
    parser.add_argument(
        '--fred-api-key',
        type=str,
        help='FRED API key for downloading real data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots as PNG files'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display plots (useful for batch processing)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_date_range(args):
    """Get start and end dates from arguments."""
    if args.days:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    else:
        start_date = args.start
        end_date = args.end
    
    return start_date, end_date


def analyze_single_bond_type(analyzer, bond_type, start_date, end_date, output_dir, args):
    """Analyze a single bond type."""
    print(f"\n{'='*60}")
    print(f"Analyzing {bond_type.upper()} Bonds")
    print(f"{'='*60}")
    
    try:
        # Generate report
        report = analyzer.generate_report(
            bond_type,
            start_date=start_date,
            end_date=end_date,
            save_data=True
        )
        
        if args.verbose:
            print(f"\nFactor Summary Statistics:")
            print(report['factors'].describe())
        
        # Create plots if requested
        if args.save_plots or not args.no_display:
            plotter = YieldCurvePlotter()
            
            # Historical factors plot
            fig = plotter.plot_historical_factors(
                report['factors'],
                bond_type.upper(),
                title=f"{bond_type.upper()} Nelson-Siegel Factors Evolution"
            )
            
            if args.save_plots:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plot_path = output_dir / f"{bond_type}_factors_{timestamp}.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved: {plot_path}")
            
            if not args.no_display:
                import matplotlib.pyplot as plt
                plt.show()
        
        return report
        
    except Exception as e:
        print(f"Error analyzing {bond_type}: {e}")
        return None


def analyze_comparison(analyzer, start_date, end_date, output_dir, args):
    """Analyze and compare Treasury and TIPS."""
    print(f"\n{'='*60}")
    print("Comparing Treasury and TIPS Bonds")
    print(f"{'='*60}")
    
    try:
        # Generate comparison report
        comparison = analyzer.compare_curves(start_date, end_date)
        
        if args.verbose:
            print(f"\nFactor Correlations:")
            for factor, corr in comparison['correlations'].items():
                print(f"  {factor}: {corr:.3f}")
            
            print(f"\nMean Factor Differences (Treasury - TIPS):")
            for factor, stats in comparison['differences'].items():
                unit = "%" if factor != "Tau" else "years"
                multiplier = 100 if factor != "Tau" else 1
                mean_diff = stats['mean_diff'] * multiplier
                print(f"  {factor}: {mean_diff:.3f} {unit}")
        
        # Create plots if requested
        if args.save_plots or not args.no_display:
            plotter = YieldCurvePlotter()
            
            # Factor comparison plot
            fig1 = plotter.plot_factor_comparison(
                comparison['treasury_factors'],
                comparison['tips_factors'],
                title="Treasury vs TIPS Nelson-Siegel Factors"
            )
            
            # Dashboard plot
            fig2 = plotter.create_summary_dashboard(comparison)
            
            if args.save_plots:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                comp_path = output_dir / f"factor_comparison_{timestamp}.png"
                dash_path = output_dir / f"dashboard_{timestamp}.png"
                
                fig1.savefig(comp_path, dpi=300, bbox_inches='tight')
                fig2.savefig(dash_path, dpi=300, bbox_inches='tight')
                
                print(f"Comparison plot saved: {comp_path}")
                print(f"Dashboard saved: {dash_path}")
            
            if not args.no_display:
                import matplotlib.pyplot as plt
                plt.show()
        
        return comparison
        
    except Exception as e:
        print(f"Error in comparison analysis: {e}")
        return None


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup
    output_dir = setup_output_directory(args.output)
    start_date, end_date = get_date_range(args)
    
    print("Nelson-Siegel Yield Curve Analysis")
    print("=" * 60)
    print(f"Bond Type: {args.bond_type}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Output Directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = YieldCurveAnalyzer(fred_api_key=args.fred_api_key)
    
    # Run analysis based on bond type
    if args.bond_type == 'both':
        result = analyze_comparison(analyzer, start_date, end_date, output_dir, args)
    else:
        result = analyze_single_bond_type(
            analyzer, args.bond_type, start_date, end_date, output_dir, args
        )
    
    if result is not None:
        print(f"\n{'='*60}")
        print("Analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("Analysis failed!")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
