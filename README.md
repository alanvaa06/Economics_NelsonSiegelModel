# Nelson-Siegel Yield Curve Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python library for yield curve modeling using the Nelson-Siegel methodology. This library provides tools for analyzing both nominal Treasury and real TIPS (Treasury Inflation-Protected Securities) yield curves with automated data fetching, parameter estimation, and visualization capabilities.

## 🌟 Features

- **Complete Nelson-Siegel Implementation**: Core model with parameter fitting and yield curve reconstruction
- **Dual Bond Type Support**: Optimized for both Treasury (nominal) and TIPS (real) yield curves  
- **Automated Data Download**: Fetch data from FRED API or use synthetic data for testing
- **Historical Analysis**: Calculate time series of Nelson-Siegel factors
- **Rich Visualizations**: Professional plotting with matplotlib for curves, factors, and comparisons
- **Modular Design**: Clean, extensible architecture following Python best practices
- **Type Hints**: Full type annotation for better development experience
- **Comprehensive Testing**: Unit tests and integration tests included

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Interactive Jupyter Analysis](#interactive-jupyter-analysis)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Data Sources](#data-sources)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Basic Installation

```bash
pip install -e .
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/economics-research/nelson-siegel-model.git
cd nelson-siegel-model

# Install with development dependencies
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[dev,interactive,data]"
```

### Dependencies

**Core Requirements:**
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation
- `scipy>=1.7.0` - Optimization and curve fitting  
- `matplotlib>=3.4.0` - Plotting and visualization
- `requests>=2.25.0` - HTTP requests

**Interactive Features:**
- `ipywidgets>=7.6.0` - Interactive notebook widgets
- `jupyter>=1.0.0` - Jupyter notebook environment
- `jupyterlab>=3.0.0` - Modern Jupyter interface

**Optional Data Sources:**
- `fredapi>=0.4.3` - Federal Reserve Economic Data API (recommended)
- `yfinance>=0.1.87` - Alternative financial data source

## 🏃‍♂️ Quick Start

### Basic Usage

```python
import numpy as np
from nelson_siegel import NelsonSiegelModel, YieldCurveAnalyzer, YieldCurvePlotter

# 1. Analyze a single yield curve
analyzer = YieldCurveAnalyzer()

# Example yield data (maturity: yield pairs)
yields_data = {
    1.0: 0.025,   # 1Y: 2.5%
    2.0: 0.028,   # 2Y: 2.8%  
    5.0: 0.032,   # 5Y: 3.2%
    10.0: 0.035,  # 10Y: 3.5%
    30.0: 0.038   # 30Y: 3.8%
}

# Analyze the curve
result = analyzer.analyze_single_curve('treasury', yields_data=yields_data)

print("Nelson-Siegel Factors:")
print(f"Level: {result['factors']['Level']:.4f}")
print(f"Slope: {result['factors']['Slope']:.4f}")  
print(f"Curvature: {result['factors']['Curvature']:.4f}")
print(f"Tau: {result['factors']['Tau']:.2f}")

# 2. Visualize the fit
plotter = YieldCurvePlotter()
fig = plotter.plot_yield_curve_fit(result)
plt.show()
```

## 🎛️ Interactive Jupyter Analysis

### Launch the Interactive Notebook

For the **most comprehensive learning experience**, use our interactive Jupyter notebook:

```bash
# Install with interactive features
pip install -e ".[interactive]"

# Launch Jupyter Lab (recommended)
jupyter lab examples/Nelson_Siegel_Interactive_Analysis.ipynb

# Or use classic Jupyter notebook
jupyter notebook examples/Nelson_Siegel_Interactive_Analysis.ipynb
```

### 📚 What's in the Interactive Notebook?

The notebook provides a **complete educational journey** through yield curve analysis:

#### **🎓 Chapter 1: Understanding Yield Curves**
- Economic fundamentals and curve shapes
- Normal, inverted, and flat curve interpretations
- Market signals and recession indicators

#### **🔬 Chapter 2: Nelson-Siegel Model Deep Dive**
- Mathematical foundation with economic meaning
- Parameter interpretation (Level, Slope, Curvature, Tau)
- Visual examples of different curve shapes

#### **🎛️ Chapter 3: Interactive Parameter Exploration**
- **Real-time sliders** for all Nelson-Siegel parameters
- **Live curve updates** as you adjust parameters
- **Treasury and TIPS** separate exploration tools
- **Market data overlay** option

#### **💡 Chapter 4: Economic Interpretation**
- **Breakeven inflation** calculations and analysis
- **Factor analysis** with economic insights
- **Investment implications** for different scenarios

#### **📈 Chapter 5: Historical Factor Analysis**
- **Interactive historical charts** with date selection
- **Economic event correlation** (2008 crisis, COVID, etc.)
- **Factor evolution patterns** and market cycles

#### **💼 Chapter 6: Practical Applications**
- **Bond relative value** analysis
- **Portfolio optimization** using factor exposures
- **Risk management** strategies

#### **🎯 Chapter 7: Advanced Strategies**
- **Factor-based investment** approaches
- **Scenario analysis** for different economic conditions
- **Custom portfolio** construction tools

### 🚀 Quick Interactive Start

```python
# In the notebook, run these cells to get started immediately:
from nelson_siegel.interactive import InteractiveYieldCurveExplorer

# Create Treasury explorer
treasury_explorer = InteractiveYieldCurveExplorer('treasury')
treasury_dashboard = treasury_explorer.create_full_interactive_dashboard()
display(treasury_dashboard)

# Create TIPS explorer  
tips_explorer = InteractiveYieldCurveExplorer('tips')
tips_dashboard = tips_explorer.create_full_interactive_dashboard()
display(tips_dashboard)
```

### 🎨 Interactive Features

- **📊 Real-time curve plotting** with parameter sliders
- **🎛️ Economic scenario testing** with preset configurations
- **📈 Historical factor visualization** with interactive timelines
- **💰 Investment strategy simulation** with portfolio tools
- **🎓 Educational content** with economic explanations

### 💻 Alternative: Google Colab

Can't install locally? **Run in Google Colab**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/economics-research/nelson-siegel-model/blob/main/examples/Nelson_Siegel_Interactive_Analysis.ipynb)

For more examples and detailed documentation, see the full documentation and examples directory.

## 🧠 Core Concepts

### Nelson-Siegel Model

The Nelson-Siegel model represents yield curves using four parameters:

```
y(τ) = β₀ + β₁[(1-e^(-τ/λ))/(τ/λ)] + β₂[((1-e^(-τ/λ))/(τ/λ)) - e^(-τ/λ)]
```

Where:
- **β₀ (Level)**: Long-term yield level
- **β₁ (Slope)**: Short-term yield component (affects short end)
- **β₂ (Curvature)**: Medium-term yield component (affects curvature)  
- **λ (Tau)**: Decay parameter controlling the maturity at which curvature peaks

### Factor Interpretation

| Factor | Economic Meaning | Effect |
|--------|------------------|---------|
| **Level** | Long-term expectations | Parallel shifts of yield curve |
| **Slope** | Short-term vs long-term | Steepening/flattening |
| **Curvature** | Medium-term dynamics | Twist around medium maturities |
| **Tau** | Decay speed | Where curvature effect is strongest |

## 📚 API Reference

### Core Classes

#### **Model Components**
- `NelsonSiegelModel`: Base model for fitting Nelson-Siegel curves
- `TreasuryNelsonSiegelModel`: Pre-configured for Treasury bonds
- `TIPSNelsonSiegelModel`: Pre-configured for TIPS bonds

#### **Analysis & Data**
- `YieldCurveAnalyzer`: High-level analysis interface
- `DataManager`: Data downloading and management
- `YieldCurvePlotter`: Visualization utilities

#### **Interactive Components** 🎛️
- `InteractiveYieldCurveExplorer`: Real-time parameter exploration
- `HistoricalFactorExplorer`: Interactive historical analysis
- `create_yield_curve_tutorial`: Educational content widget

### Quick Reference

```python
# Basic model usage
from nelson_siegel import TreasuryNelsonSiegelModel
model = TreasuryNelsonSiegelModel()
model.fit(maturities, yields)
factors = model.get_factors()

# Interactive exploration (Jupyter)
from nelson_siegel.interactive import InteractiveYieldCurveExplorer
explorer = InteractiveYieldCurveExplorer('treasury')
dashboard = explorer.create_full_interactive_dashboard()
display(dashboard)

# High-level analysis
from nelson_siegel import YieldCurveAnalyzer
analyzer = YieldCurveAnalyzer()
result = analyzer.analyze_single_curve('treasury', yields_data=data)
```

See the **Interactive Jupyter Notebook** and examples directory for detailed usage.

## 📖 Examples

### 🎛️ **Interactive Jupyter Notebook** (Recommended)
**`examples/Nelson_Siegel_Interactive_Analysis.ipynb`**
- Comprehensive educational journey with 8 chapters
- Real-time parameter exploration with sliders
- Economic interpretation and investment applications
- Historical factor analysis with interactive charts

### 🐍 **Python Scripts**
**`examples/basic_usage.py`**
- Quick start examples and API demonstrations
- Static analysis and visualization examples

**`examples/legacy_nelson_siegel.py`**
- Migration guide from original script
- Recreates original functionality with new architecture

### 🤖 **Command Line Interface**
**`scripts/run_analysis.py`**
- Batch processing and automated analysis
- Configurable parameters and output formats

```bash
# Run Treasury analysis for last year
python scripts/run_analysis.py treasury --days 365 --save-plots

# Compare Treasury and TIPS
python scripts/run_analysis.py both --start 2020-01-01 --end 2023-12-31
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/economics-research/nelson-siegel-model.git
cd nelson-siegel-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/nelson_siegel --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Type checking
mypy src/nelson_siegel
```

## 🤝 Contributing

We welcome contributions! Please see the development section for setup instructions and follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests and documentation
4. Run quality checks
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/economics-research/nelson-siegel-model/issues)
- **Email**: research@example.com

## 🙏 Acknowledgments

- Nelson, C. R. and A. F. Siegel (1987). "Parsimonious Modeling of Yield Curves". Journal of Business, 60(4), 473-489.
- Federal Reserve Bank of St. Louis for providing FRED API access