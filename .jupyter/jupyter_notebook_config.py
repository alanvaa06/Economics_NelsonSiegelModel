# Jupyter Notebook Configuration for Nelson-Siegel Interactive Analysis

c = get_config()

# Enable widgets
c.NotebookApp.nbserver_extensions = {
    'jupyter_nbextensions_configurator': True,
}

# Widget configuration
c.InteractiveShellApp.extensions = ['ipywidgets']

# Display configuration
c.InlineBackend.figure_formats = {'svg', 'png'}
c.InlineBackend.rc = {
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
}

# Security settings for interactive widgets
c.NotebookApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self' *"
    }
}
