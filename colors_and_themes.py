import dash_bootstrap_components as dbc
import plotly.io as pio
from dash import html


#======SETUP COLORS AND LIGHT/DARK THEMES===============

color_map = {'EQ101': '#636efa', 'EQ102': '#ef553b', 'EQ103': '#00cc96', 'EQ104': '#ab63fa'}

table_font_size = 10

# Create a bright and dark theme for the dash app. The theme is used for the tables and page background

# Create a custom dark theme for the charts. The color will match the dark theme color below
custom_template = pio.templates['plotly_dark'] 

theme_bright = dbc.themes.SANDSTONE
theme_chart_bright = pio.templates['seaborn']  # available plotly themes: simple_white, plotly, plotly_dark, ggplot2, seaborn, plotly_white, none

# available dash bootstrap dbc themes: BOOTSTRAP, CERULEAN, COSMO, CYBORG, DARKLY, FLATLY, JOURNAL, LITERA, LUMEN, LUX, MATERIA, MINTY, MORPH, PULSE, QUARTZ, SANDSTONE, SIMPLEX, SKETCHY, SLATE, SOLAR, SPACELAB, SUPERHERO, UNITED, VAPOR, YETI, ZEPHYR
darktheme = "SUPERHERO"
if darktheme == "SUPERHERO":
    theme_dark = dbc.themes.SUPERHERO
    custom_template.layout.paper_bgcolor = '#0f2537' # match the SUPERHERO theme
    custom_template.layout.plot_bgcolor = '#ced4da'  # light gray plot background color
elif darktheme == "SOLAR":
    theme_dark = dbc.themes.SOLAR
    custom_template.layout.paper_bgcolor = '#002b36' # match the SOLAR theme
    custom_template.layout.plot_bgcolor = '#ced4da'  # light gray plot background color
elif darktheme == "SLATE":
    theme_dark = dbc.themes.SLATE
    custom_template.layout.paper_bgcolor = '#272b30' # match the SOLAR theme
    custom_template.layout.plot_bgcolor = '#ced4da'  # light gray plot background color
elif darktheme == "DARKLY":
    theme_dark = dbc.themes.DARKLY
    custom_template.layout.paper_bgcolor = '#222222' # match the SOLAR theme
    custom_template.layout.plot_bgcolor = '#ced4da'  # light gray plot background color

pio.templates['custom_dark'] = custom_template
theme_chart_dark = pio.templates['custom_dark']

footer = html.P(
                [
                    html.Span('Created by Sean Trautman  ', className='mr-2'),
                    html.A(html.I(className='fas fa-envelope-square mr-1'), href='mailto:sean.trautman@intel.com')
                ], 
                className='lead')

if __name__ == '__main__':
    print(theme_dark)