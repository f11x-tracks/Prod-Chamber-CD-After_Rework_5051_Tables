df_chamber.to_csv('data-chamber.csv', index=False)
df_cd.to_csv('data-cd.csv', index=False)
df_limits.to_csv('data-limits.csv', index=False)
#------------------------------------------------------------#

df_limits = df_limits.rename(columns={'PROCESS_OPERATION': 'OPN'})
df_limits['OPN'] = df_limits['OPN'].astype(str)
df_limits = df_limits.rename(columns={'LO_CONTROL_LMT': 'LCL'})
df_limits = df_limits.rename(columns={'UP_CONTROL_LMT': 'UCL'})
df_limits[['LAYER', 'STRUCTURE', 'STAT']] = df_limits['RAW_PARAMETER_NAME'].str.split(';', expand=True)
# STAT is WAFER_SIGMA, WAFER_MEAN, and PERCENT_MEASURED
df_limits = df_limits[df_limits['STAT'] != 'PERCENT_MEASURED'] # Remove PERCENT_MEASURED
df_limits['TECH'] = df_limits['ROUTE'].str[:4].str[-2:]
df_limits = df_limits.drop(columns=['RAW_PARAMETER_NAME', 'PRODUCT', 'ROUTE'])
grouped_df_limits = df_limits.groupby(['LAYER', 'STRUCTURE', 'TECH', 'OPN'])
# Create new columns to reshape the UCL, LCL from tall to wide format
df_limits['WAFER_SIGMA_LCL'] = None
df_limits['WAFER_SIGMA_UCL'] = None
df_limits['WAFER_MEAN_LCL'] = None
df_limits['WAFER_MEAN_UCL'] = None
# Move values based on STAT
df_limits.loc[df_limits['STAT'] == 'WAFER_SIGMA', 'WAFER_SIGMA_LCL'] = df_limits['LCL']
df_limits.loc[df_limits['STAT'] == 'WAFER_SIGMA', 'WAFER_SIGMA_UCL'] = df_limits['UCL']
df_limits.loc[df_limits['STAT'] == 'WAFER_MEAN', 'WAFER_MEAN_LCL'] = df_limits['LCL']
df_limits.loc[df_limits['STAT'] == 'WAFER_MEAN', 'WAFER_MEAN_UCL'] = df_limits['UCL']
# Drop the original LCL and UCL columns
df_limits = df_limits.drop(columns=['LCL', 'UCL', 'STAT'])
# Remove duplicate rows
df_limits = df_limits.drop_duplicates()
# Replace None with NaN in the specified columns
columns_to_replace = ['WAFER_SIGMA_LCL', 'WAFER_SIGMA_UCL', 'WAFER_MEAN_LCL', 'WAFER_MEAN_UCL']
df_limits.loc[:, columns_to_replace] = df_limits[columns_to_replace].replace({None: np.nan}).infer_objects(copy=False)
# Group by TECH, OPN, STRUCTURE and aggregate the other columns
df_limits = df_limits.groupby(['TECH', 'OPN', 'SPC_OPERATION', 'STRUCTURE']).agg({
    'WAFER_SIGMA_LCL': 'mean',
    'WAFER_SIGMA_UCL': 'mean',
    'WAFER_MEAN_LCL': 'mean',
    'WAFER_MEAN_UCL': 'mean',
    'LAYER': 'first'  # Assuming LAYER is the same for identical TECH, OPN, STRUCTURE
}).reset_index()


# Split the 'RAW_PARAMETER_NAME' column into multiple columns
df_cd[['PARAM1', 'LAYER', 'STRUCTURE', 'PARAM4']] = df_cd['RAW_PARAMETER_NAME'].str.split(';', expand=True)
df_cd = df_cd.drop(columns=['RAW_PARAMETER_NAME', 'PARAM1', 'PARAM4'])
df_cd = df_cd.rename(columns={'PROCESS_OPERATION': 'OPN'})
df_cd['OPN'] = df_cd['OPN'].astype(str)
df_cd = df_cd.rename(columns={'RAW_WAFER3': 'WAF3'})
df_cd = df_cd.rename(columns={'ENTITY': 'Tool'})
df_cd['TECH'] = df_cd['ROUTE'].str[:4].str[-2:]
# Remove rows where 'CD' is 0
df_cd = df_cd[df_cd['RAW_VALUE'] != 0]
# Convert RAW_VALUE to float64, non-convertible values will be set to NaN
df_cd['RAW_VALUE'] = pd.to_numeric(df_cd['RAW_VALUE'], errors='coerce')
# drop rows with NaN values after converting RAW_VALUE to float64
df_cd = df_cd.dropna(subset=['RAW_VALUE'])

df_chamber = df_chamber.rename(columns={'LOT1': 'LOT'})
df_chamber = df_chamber.rename(columns={'OPERATION1': 'OPN'})
df_chamber['OPN'] = df_chamber['OPN'].astype(str)
df_chamber = df_chamber.rename(columns={'LOT_ROUTE': 'ROUTE'})
df_chamber['TECH'] = df_chamber['ROUTE'].str[:4].str[-2:]
df_chamber = df_chamber.sort_values(by=['OPN', 'LOT', 'WAF3', 'START_DATE'], ascending=[True, True, True, True])
# Merge df_limits with df_cd by matching TECH, OPN, STRUCTURE
df_cd = pd.merge(df_cd, df_limits, on=['TECH', 'OPN', 'STRUCTURE'], how='left')

#create columns for chamber data to go from tall to wide format
bake_num = 3
chamberl = ['ADH', 'COT', 'ITC', 'BCT', 'PCT', 'DEV']
ch_columns = [f'{stage}-CH' for stage in chamberl + [f'BAKE{i}' for i in range(1, bake_num + 1)]]
for col in ch_columns:
    df_chamber[col] = None

# Flatten the CHAMBER column
def flatten_chamber(group):
    adh = cot = itc = bct = pct = dvlp = None
    bake = [None] * bake_num
    bake_count = 0
    for idx, row in group.iterrows():
        ch = row['CHAMBER']
        if row['REWORK'] == REWORK_FLAG_Y or row['REWORK'] == REWORK_FLAG_N:
            if ch.startswith(('ADH', 'CADH')):
                adh = ch
            elif ch.startswith('COT'):
                cot = ch
            elif ch.startswith('ITC'):
                itc = ch
            elif ch.startswith('BCT'):
                bct = ch
            elif ch.startswith('PCT'):
                pct = ch
            elif ch.startswith('DEV'):
                dvlp = ch
            elif ch.startswith(('RGCH', 'CGCH', 'CPHG', 'CPHP', 'PHP')):
                if bake_count < 3:
                    bake[bake_count] = ch
                    bake_count += 1
    group['ADH-CH'] = adh
    group['COT-CH'] = cot
    group['ITC-CH'] = itc
    group['BCT-CH'] = bct
    group['PCT-CH'] = pct
    group['DEV-CH'] = dvlp
    for i in range(bake_num):
        group[f'BAKE{i+1}-CH'] = bake[i]
    return group.iloc[0]

df_chamber = df_chamber.groupby(['OPN', 'LOT', 'WAF3'], group_keys=False).apply(flatten_chamber).reset_index(drop=True)
df_chamber = df_chamber.drop(columns=['CHAMBER'])

split_columns = df_chamber['LOT_RECIPE'].str.split(' ', n=1, expand=True)
df_chamber['RESIST_RCP'] = split_columns[0]
df_chamber['SCANNER'] = split_columns[1]
df_chamber = df_chamber.drop(columns=['LOT_RECIPE'])
df_chamber['RESIST'] = df_chamber['RESIST_RCP'].str.split('-').str[1]
df_chamber['RESIST'] = df_chamber['RESIST'].astype(str)

df_cd = pd.merge(df_cd, df_chamber, on=['LOT', 'OPN', 'WAF3'], how='left')
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 100)
#print(df_chamber)
#print(df_limits)
#print(df_cd)
#df_chamber.to_excel('output.xlsx', index=False)

# clean up after merging the dataframes
df = df_cd.drop(columns=['STANDARD_FLAG', 'VALID_FLAG', 'MEASUREMENT_SET_NAME', 'ROUTE_y', 'TECH_y', 'LAYER_y', 'SPC_OPERATION_y', 'LOT_ENTITY', 'STATE'])
df = df.rename(columns={'WAFER_MEAN_UCL':'UCL','WAFER_MEAN_LCL':'LCL','WAFER_SIGMA_UCL':'SIGMA_UCL','WAFER_SIGMA_LCL':'SIGMA_LCL', 'RAW_VALUE':'CD', 'NATIVE_X_COL':'X-COL', 'NATIVE_Y_ROW':'Y-ROW', 'LOT_RETICLE':'RETICLE', 'LOT_PROCESSED_WAFER_COUNT':'QTY', 'LOT_LOAD_PORT':'PORT', 'LOT_ABORT_FLAG':'ABORT','TECH_x': 'TECH', 'LAYER_x': 'LAYER', 'WAF3': 'WFR', 'ROUTE_x': 'ROUTE', 'SPC_OPERATION_x': 'SPC_OPN', 'ENTITY_DATA_COLLECT_DATE': 'DATETIME'})
df['X-Y'] = df['X-COL'].astype(str) + '-' + df['Y-ROW'].astype(str)
df['MP'] = None
# Assign MP values based on unique STRUCTURE values for each TECH, OPN grouping
def assign_mp(group):
    unique_structures = group['STRUCTURE'].unique()
    structure_map = {structure: f'MP{i+1}' for i, structure in enumerate(unique_structures)}
    group['MP'] = group['STRUCTURE'].map(structure_map)
    return group

df = df.groupby(['TECH', 'OPN']).apply(assign_mp).reset_index(drop=True)

df['TECH_LAYER'] = df['TECH'] + '_' + df['LAYER']

# Group by TECH_LAYER and aggregate unique values for Tool, LOT, and WFR
grouped = df.groupby('TECH_LAYER').agg({
    'Tool': lambda x: list(x.unique()),
    'LOT': lambda x: list(x.unique()),
    'WFR': lambda x: list(x.unique())
}).reset_index()

# Convert the grouped DataFrame to a dictionary for easier access
tech_layer_dict = grouped.set_index('TECH_LAYER').T.to_dict('list')

# Example of how to access the unique values for a specific TECH_LAYER
# unique_entities = tech_layer_dict['TECH_LAYER_VALUE'][0]
# unique_lots = tech_layer_dict['TECH_LAYER_VALUE'][1]
# unique_wfrs = tech_layer_dict['TECH_LAYER_VALUE'][2]

unique_tech_layer = df['TECH_LAYER'].unique()

df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%Y-%m-%d %H:%M:%S')
#df['DATETIME'] = df['DATETIME'].dt.tz_localize('UTC')
df['DATETIME'] = pd.to_datetime((df['DATETIME']))
df['DATE'] = df['DATETIME'].dt.normalize()

grouped = df.groupby(['TECH', 'LAYER', 'MP'])['CD']
mean = grouped.mean()
std = grouped.std()
ave_p3s = (mean + 3 * std).round(1)
ave_m3s = (mean - 3 * std).round(1)

# Reset index to avoid ambiguity during merge
df = df.reset_index(drop=True)

# Merge ave_p3s and ave_m3s back to the original DataFrame
df = df.merge(ave_p3s.rename('ave_p3s'), on=['TECH', 'LAYER', 'MP'])
df = df.merge(ave_m3s.rename('ave_m3s'), on=['TECH', 'LAYER', 'MP'])

# Fill NaN, None, or blank values in UCL and LCL
df['UCL'] = df['UCL'].replace('', np.nan).fillna(df['ave_p3s']).infer_objects(copy=False)
df['LCL'] = df['LCL'].replace('', np.nan).fillna(df['ave_m3s']).infer_objects(copy=False)

# Drop the temporary columns if not needed
df.drop(columns=['ave_p3s', 'ave_m3s'], inplace=True)

df.to_excel('cd_df.xlsx', index=False)

#===START DASH AND CREATE LAYOUT OF TABLES/GRAPHS===============
#======SETUP COLORS AND LIGHT/DARK THEMES===============

color_map = {'EQ101': '#636efa', 'EQ102': '#ef553b', 'EQ103': '#00cc96', 'EQ104': '#ab63fa'}

table_font_size = 10

OOS_highlight = '#cfb974'

# Create a bright and dark theme for the dash app. The theme is used for the tables and page background

theme_bright = dbc.themes.SANDSTONE
theme_chart_bright = pio.templates['seaborn']  # available plotly themes: simple_white, plotly, plotly_dark, ggplot2, seaborn, plotly_white, none

'''
# Create a custom dark theme for the charts. The color will match the dark theme color below
custom_template = pio.templates['plotly_dark']  # was plotly_dark
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
'''
tukey_table_cell_highlight = '#cfb974'  # Color for highlighting a tukeyHSD flagged cell

# background color -> #002b36    # gray dark
# 'backgroundColor': 'rgb(40, 40, 40)' # Table header color
# 'backgroundColor': '#052027',         # Color for highlighting a tukeyHSD flagged cell
# radio button color -> #b58900  # dark gold
# slider bar color  -> #cfb974   # light gold

#===START DASH AND CREATE LAYOUT OF TABLES/GRAPHS================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE]) # external_stylesheets=[theme_bright]

title = html.H1("PROCESS DASHBOARD", style={'font-size': '18px'}) #'color': 'white',

#theme_switch = theme_dark
chart_theme = theme_chart_bright # or theme_chart_dark. Update app=dash stylesheet too

max_table_rows = 30

dropdown_options = [{'label': value, 'value': value} for value in unique_tech_layer]

tech_layer_dd = html.Div([
        dcc.Dropdown(
            id='tech_layer_dd',
            options=dropdown_options,
            value='51_ODT'  # Default value
        )])

chart_MP_radio = html.Div(
    dcc.RadioItems(
        id='chart_MP_radio', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                 for x in ['MP1', 'MP2', 'MP3', 'MP4', 'MP5']],  # radio button labels and values
        value='MP1',   # Default
        labelStyle={'display': 'inline-block'}
        ))

chart_range_slider = html.Div([dcc.RangeSlider(
        id = 'limit-slider',
        min = 100,         # Initial value, will be updated by callback
        max = 200,        # Initial value, will be updated by callback
        step = 10,      # Initial value, will be updated by callback
        value=[110, 190],  # Initial value, will be updated by callback
        tooltip={"placement": "bottom", "always_visible": False})])

calendar = html.Div(["Date Range ",
        dcc.DatePickerRange(
            id="date-range",
            min_date_allowed=df['DATE'].min().date(),
            max_date_allowed=df['DATE'].max().date(),
            start_date=df['DATE'].min().date(),
            end_date=df['DATE'].max().date(),
        )])

summary_tableh = html.Div(["Summary Table", dt.DataTable(id='summary-table',
        columns=[
            {'name': ['Summary Table', 'No'], 'id': 'no_rows'},
            {'name': ['Summary Table','Tool'], 'id': 'tool'},
            {'name': ['Summary Table','Chamber'], 'id': 'chamber'},
            {'name': ['Summary Table','Count'], 'id': 'count'},
            {'name': ['Summary Table','Mean'], 'id': 'mean'},
            {'name': ['Summary Table','Sigma'], 'id': 'sigma'}
        ],
        data=[{'no_rows': i} for i in range(1,max_table_rows)],
        sort_action='native',
        sort_mode='multi',
        editable=False,
        merge_duplicate_headers=True,
        style_cell={'textAlign': 'center'},
        style_header={          
            'fontWeight': 'bold'
        })
        ])  # style={'display': 'inline-table', 'margin':'10px', 'width': '20%'}

tool_checklist = html.Div(dcc.Checklist(
        id="tool_list",  # id names will be used by the callback to identify the components
        options=[],  # List of tools updated by the callback
        value=[],  
        inline=True))

chamber_list_radio = html.Div(
        dcc.RadioItems(
        id='chamber', 
        options=[{'value': x, 'label': x[:-3]}  
                for x in ch_columns],  
        value='DEV-CH',   # Default
        labelStyle={'display': 'inline-block'}
        ))

line_chart1 = html.Div([dcc.Graph(figure={}, id='line-chart1')])  # figure is blank dict because created in callback below

boxplot1 = html.Div([dcc.Graph(figure={}, id='box-plot1')])

line_chart2 = html.Div([dcc.Graph(figure={}, id='line-chart2')]) 
line_chartXcol = html.Div([dcc.Graph(figure={}, id='line_chartXcol')]) 
line_chartYrow = html.Div([dcc.Graph(figure={}, id='line_chartYrow')]) 

boxplot_lot = html.Div([dcc.Graph(figure={}, id='boxplot_lot')])
line_chartXcol_lot = html.Div([dcc.Graph(figure={}, id='line_chartXcol_lot')]) 
line_chartYrow_lot = html.Div([dcc.Graph(figure={}, id='line_chartYrow_lot')]) 

# Group by LOT, OPN, and MP and calculate mean and std of CD
df_lot_summary = df.groupby(['LOT', 'LAYER', 'MP']).agg(
    STRUCTURE=('STRUCTURE', 'first'),
    Tool=('Tool', 'first'),
    DATETIME=('DATETIME', 'first'),
    CD_mean=('CD', 'mean'),
    CD_std=('CD', 'std'),
    SIGMA_UCL=('SIGMA_UCL', 'first'),
    REWORK=('REWORK', 'first'),
    RESIST_RCP=('RESIST_RCP', 'first'),
    QTY=('QTY', 'first'),
    SPC_OPN=('SPC_OPN', 'first'),
    ROUTE=('ROUTE', 'first'),
    ABORT=('ABORT', 'first'),
    OPN=('OPN', 'first'),
    SCANNER=('SCANNER', 'first'),
    TECH_LAYER=('TECH_LAYER', 'first'),
    DATE=('DATE', 'first'),
).reset_index()
df_lot_summary['CD_mean'] = df_lot_summary['CD_mean'].round(1)
df_lot_summary['CD_std'] = df_lot_summary['CD_std'].round(1)

lot_list_table = html.Div([
    html.Button("Unselect All", id="unselect-button"),
    dt.DataTable(
        id='lot_list_table',
        columns=[{'name': col, 'id': col} for col in df_lot_summary.columns],
        data=df_lot_summary.to_dict('records'),
        style_table={'height': '200px', 'overflowX': 'auto', 'overflowY': 'auto'},
        row_selectable='multi',
        fixed_rows={'headers': True, 'data': 0},
        sort_action='native',
        sort_mode='multi',
        filter_action='native',  # Enable column filtering
        style_cell={'textAlign': 'center'},
        tooltip_header={col: {'value': col, 'type': 'markdown'} for col in df_lot_summary.columns}
    )])


# Layout of the dash graphs, tables, drop down menus, etc
# Using dbc container for styling/formatting
app.layout = dbc.Container([
    # Summary and Tukey Tables
    dbc.Row([
        dbc.Col(calendar, width={"size":5, "justify":"left"}),
        dbc.Col(tech_layer_dd, width={"size":3, "justify":"between"}),
        dbc.Col(chart_MP_radio, width={"size":4})]),
    # Charts and Boxplot
    dbc.Row([
        dbc.Col(chart_range_slider, width={"size":12})]),
    dbc.Row([
        dbc.Col(tool_checklist, width={"size":6}),
        dbc.Col(chamber_list_radio, width={"size":6})]),
    dbc.Row([
        dbc.Col(line_chart1, width={"size":6}),
        dbc.Col(boxplot1, width={"size":6})]),
    dbc.Row([
        dbc.Col(line_chart2, width={"size":6}),
        dbc.Col(line_chartXcol, width={"size":3}),
        dbc.Col(line_chartYrow, width={"size":3})]),
    dbc.Row([
        dbc.Col(lot_list_table, width={"size":12})]),
    dbc.Row([
        dbc.Col(boxplot_lot, width={"size":6}),
        dbc.Col(line_chartXcol_lot, width={"size":3}),
        dbc.Col(line_chartYrow_lot, width={"size":3})]),
    dbc.Row([
        dbc.Col(summary_tableh, width={"size":4})]),
    ], fluid=True, className="dbc dbc-row-selectable")

#=====CREATE INTERACTIVE GRAPHS=============
# Callbacks are used to update the graphs and tables when the user changes the inputs 
# chart upper lower limit slider
@app.callback(
    Output('limit-slider', 'min'),
    Output('limit-slider', 'max'),
    Output('limit-slider', 'step'),
    Output('limit-slider', 'value'),
    Input('tech_layer_dd', 'value'),
    Input('chart_MP_radio', 'value')
)
def update_slider(selected_tech_layer, selected_chart_mpx):
    # Filter the DataFrame based on selected_tech_layer and selected_chart_mpx
    filtered_df = df[(df['TECH_LAYER'] == selected_tech_layer) & (df['MP'] == selected_chart_mpx)]
    # Ensure there is data after filtering
    if filtered_df.empty:
        raise ValueError("No data found for the selected TECH_LAYER and MP combination.")
    # Get the LCL and UCL values
    lcl = filtered_df['LCL'].values[0]
    ucl = filtered_df['UCL'].values[0]
    # Calculate the limits and step
    lower_limit = lcl - (0.1 * abs(lcl))
    upper_limit = ucl + (0.1 * abs(ucl))
    step = round((upper_limit - lower_limit) / 40, 1)
    value = [lcl, ucl]
    return lower_limit, upper_limit, step, value

# Define the callback to update the tool_checklist based on which tools have been selected
@app.callback(
    Output('tool_list', 'options'),
    Input('tech_layer_dd', 'value')
)
def update_tool_checklist(selected_tech_layer):
    # tech_layer_dict is a dictionary with keys as tech_layer values
    # and values as lists of tool options
    tool_list = tech_layer_dict.get(selected_tech_layer, [])
    return tool_list[0] if tool_list else []

# Summary table update 
@app.callback(
    Output('summary-table', 'data'),     # args are component id and then component property. component property is passed
    Input('date-range', 'start_date'),  # in order to the chart function below
    Input('date-range', 'end_date'),
    State('summary-table', 'data'),
    Input('tech_layer_dd', 'value'),
    Input('chart_MP_radio', 'value'),
    Input('chamber', 'value'))  # Add chamber input
def summary_table(start_date, end_date, rows, selected_tech_layer, selected_chart_mpx, selected_chamber):
    # Filter the data
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")
    
    # Group by Tool and selected_chamber and aggregate
    dfsummary = filtered_data.groupby(['Tool', selected_chamber]).agg(
        count=('CD', 'size'),
        mean=('CD', 'mean'),
        std=('CD', 'std')
    ).reset_index()
    
    # Format mean and std columns
    dfsummary['mean'] = dfsummary['mean'].map('{:.1f}'.format)
    dfsummary['std'] = dfsummary['std'].map('{:.1f}'.format)
    
    # Map summary data to rows
    summaryd = {'tool': 'Tool', 'chamber': selected_chamber, 'count': 'count', 'mean': 'mean', 'sigma': 'std'}
    for i, row in enumerate(rows):
        for key, value in summaryd.items():
            try:
                row[key] = dfsummary.at[i, value]
            except KeyError:
                row[key] = ''
    
    return rows

# Lot list table
@app.callback(
    Output('lot_list_table', 'data'),     # args are component id and then component property. component property is passed
    Input('date-range', 'start_date'),  # in order to the chart function below
    Input('date-range', 'end_date'),
    Input('tech_layer_dd', 'value'),
    Input('chart_MP_radio', 'value'))
def lot_list_table(start_date, end_date, selected_tech_layer, selected_chart_mpx):
    filtered_data = df_lot_summary.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx") 
    return filtered_data.to_dict('records')

# Create plotly express line chart 1
@app.callback(
    Output("line-chart1", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"))  # Add MP dropdown as input
def update_line_chart(tool, start_date, end_date, limits, selected_tech_layer, selected_chart_mpx):    # callback function arg 'tool' refers to the component property of the input or "value" above
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")  # If table turned ON then update with values
    tooll = sorted(filtered_data['Tool'].unique().tolist())
    mask = filtered_data.Tool.isin(tool) 
    layer = filtered_data['LAYER'].iloc[0]
    resist = str(filtered_data['RESIST'].iloc[0])
    opn = str(filtered_data['OPN'].iloc[0])
    structure = filtered_data['STRUCTURE'].iloc[0]
    title = layer + ' (' + resist + '/' + opn + ') ' + structure if not filtered_data.empty else "No Data"                                  # Create a panda series with True/False of only tools selected 
    fig = px.line(filtered_data[mask],   
        x='DATETIME', y='CD', color='Tool'
        ,category_orders={'Tool':tooll}  # can manually set colors color_discrete_sequence = ['darkred', 'dodgerblue', 'green', 'tan']
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR', 'RESIST_RCP', 'PRODUCT', 'RETICLE', 'SLOT', 'ITC-CH', 'BCT-CH', 'COT-CH', 'PCT-CH', 'BAKE1-CH', 'BAKE1-CH','BAKE2-CH','DEV-CH','BAKE3-CH','X-Y']
        ,markers=True,
        title=title,
        template=chart_theme)
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express box plot
@app.callback(
    Output("box-plot1", "figure"), 
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input('tech_layer_dd', 'value'),
    Input('chart_MP_radio', 'value'))
def generate_bx_chamber(start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx):
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")  # If table turned ON then update with values
    tooll = sorted(filtered_data['Tool'].unique().tolist())
    sorted_chambers = sorted(filtered_data[chamber].unique().tolist())
    layer = filtered_data['LAYER'].iloc[0]
    resist = str(filtered_data['RESIST'].iloc[0])
    opn = str(filtered_data['OPN'].iloc[0])
    structure = filtered_data['STRUCTURE'].iloc[0]
    title = layer + ' (' + resist + '/' + opn + ') ' + structure if not filtered_data.empty else "No Data"
    fig = px.box(filtered_data, x="Tool", y='CD', color=chamber, notched=True, template=chart_theme, hover_data=[filtered_data['LOT'], filtered_data['WFR'],  filtered_data['RESIST_RCP']], category_orders={"Tool": tooll, chamber: sorted_chambers}, title=title)
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express box plot for lot selections
@app.callback(
    Output("boxplot_lot", "figure"), 
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input('tech_layer_dd', 'value'),
    Input('chart_MP_radio', 'value'),
    Input('lot_list_table', 'selected_rows'),
    State("lot_list_table", "data"))
def generate_bx_chamber(start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx, selected_rows, data):
    if selected_rows:
        selected_lots = [data[i]['LOT'] for i in selected_rows]
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx and LOT in @selected_lots")
    else:
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")
    tooll = sorted(filtered_data['Tool'].unique().tolist())
    sorted_chambers = sorted(filtered_data[chamber].unique().tolist())
    layer = filtered_data['LAYER'].iloc[0]
    resist = str(filtered_data['RESIST'].iloc[0])
    opn = str(filtered_data['OPN'].iloc[0])
    structure = filtered_data['STRUCTURE'].iloc[0]
    title = layer + ' (' + resist + '/' + opn + ') ' + structure if not filtered_data.empty else "No Data"
    fig = px.box(filtered_data, x="Tool", y='CD', color=chamber, notched=True, template=chart_theme, hover_data=[filtered_data['LOT'], filtered_data['WFR'],  filtered_data['RESIST_RCP']], category_orders={"Tool": tooll, chamber: sorted_chambers}, title=title)
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Callback to unselect all rows from the lot table
@app.callback(
    Output('lot_list_table', 'selected_rows'),
    Input('unselect-button', 'n_clicks')
)
def unselect_all(n_clicks):
    if n_clicks:
        return []
    return dash.no_update

# Create plotly express line chart 2 - CD by chamber
@app.callback(
    Output("line-chart2", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"))  # Add MP dropdown as input
def update_line_chart(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx):    # callback function arg 'tool' refers to the component property of the input or "value" above
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")  # If table turned ON then update with values
    tooll = sorted(filtered_data['Tool'].unique().tolist())
    mask = filtered_data.Tool.isin(tool) 
    layer = filtered_data['LAYER'].iloc[0]
    resist = str(filtered_data['RESIST'].iloc[0])
    opn = str(filtered_data['OPN'].iloc[0])
    structure = filtered_data['STRUCTURE'].iloc[0]
    title = layer + ' (' + resist + '/' + opn + ') ' + structure if not filtered_data.empty else "No Data"                                  # Create a panda series with True/False of only tools selected 
    fig = px.line(filtered_data[mask],   
        x='DATETIME', y='CD', color=chamber
        ,category_orders={'Tool':tooll}  # can manually set colors color_discrete_sequence = ['darkred', 'dodgerblue', 'green', 'tan']
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR', 'RESIST_RCP', 'PRODUCT', 'RETICLE', 'SLOT', 'ITC-CH', 'BCT-CH', 'COT-CH', 'PCT-CH', 'BAKE1-CH', 'BAKE1-CH','BAKE2-CH','DEV-CH','BAKE3-CH','X-Y']
        ,markers=True,
        title=title,
        template=chart_theme)
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express by Xcol
@app.callback(
    Output("line_chartXcol", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"))  # Add MP dropdown as input
def update_line_chart(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx):    # callback function arg 'tool' refers to the component property of the input or "value" above
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")  # If table turned ON then update with values
    mask = filtered_data.Tool.isin(tool) 
    sorted_data = filtered_data[mask].sort_values(by='X-COL')
    #tooll = sorted(filtered_data['Tool'].unique().tolist())
    title = "Across Lots"                                  
    fig = px.line(sorted_data,   
        x='X-COL', y='CD', color=chamber
        #,category_orders={'Tool':tooll}  # can manually set colors color_discrete_sequence = ['darkred', 'dodgerblue', 'green', 'tan']
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR']
        ,markers=True,
        title=title,
        template=chart_theme)
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express by Yrow
@app.callback(
    Output("line_chartYrow", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"))  # Add MP dropdown as input
def update_line_chart(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx):    # callback function arg 'tool' refers to the component property of the input or "value" above
    filtered_data = df.query("DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")  # If table turned ON then update with values
    mask = filtered_data.Tool.isin(tool) 
    sorted_data = filtered_data[mask].sort_values(by='Y-ROW')
    #tooll = sorted(filtered_data['Tool'].unique().tolist())
    title = "Across Wafer"                                  
    fig = px.line(sorted_data,   
        x='Y-ROW', y='CD', color=chamber
        #,category_orders={'Tool':tooll}  # can manually set colors color_discrete_sequence = ['darkred', 'dodgerblue', 'green', 'tan']
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR']
        ,markers=True,
        title=title,
        template=chart_theme)
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express by Xcol for lot selection
@app.callback(
    Output("line_chartXcol_lot", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"),
    Input('lot_list_table', 'selected_rows'),
    State("lot_list_table", "data"))
def update_line_chart(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx, selected_rows, data):    # callback function arg 'tool' refers to the component property of the input or "value" above
    if selected_rows:
        selected_lots = [data[i]['LOT'] for i in selected_rows]
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx and LOT in @selected_lots")
    else:
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")
    mask = filtered_data.Tool.isin(tool) 
    sorted_data = filtered_data[mask].sort_values(by='X-COL')
    #tooll = sorted(filtered_data['Tool'].unique().tolist())
    title = "Selected Lots"                                  
    fig = px.line(sorted_data,   
        x='X-COL', y='CD', color=chamber
        #,category_orders={'Tool':tooll}  # can manually set colors color_discrete_sequence = ['darkred', 'dodgerblue', 'green', 'tan']
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR']
        ,markers=True,
        title=title,
        template=chart_theme)
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express by Yrow for lot selection
@app.callback(
    Output("line_chartYrow_lot", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chamber", "value"),
    Input("limit-slider", "value"),
    Input("tech_layer_dd", "value"),  # Add TECH_LAYER dropdown as input
    Input("chart_MP_radio", "value"),
    Input('lot_list_table', 'selected_rows'),
    State("lot_list_table", "data"))
def update_line_chart(tool, start_date, end_date, chamber, limits, selected_tech_layer, selected_chart_mpx, selected_rows, data):    # callback function arg 'tool' refers to the component property of the input or "value" above
    if selected_rows:
        selected_lots = [data[i]['LOT'] for i in selected_rows]
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx and LOT in @selected_lots")
    else:
        filtered_data = df.query(
            "DATE >= @start_date and DATE <= @end_date and TECH_LAYER == @selected_tech_layer and MP == @selected_chart_mpx")
    mask = filtered_data.Tool.isin(tool) 
    sorted_data = filtered_data[mask].sort_values(by='Y-ROW')
    #tooll = sorted(filtered_data['Tool'].unique().tolist())
    title = "Across Wafer"                                  
    fig = px.line(sorted_data,   
        x='Y-ROW', y='CD', color=chamber
        #,category_orders={'Tool':tooll}  # can manually set colors color_discrete_sequence = ['darkred', 'dodgerblue', 'green', 'tan']
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['LOT', 'WFR']
        ,markers=True,
        title=title,
        template=chart_theme)
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[0] - 0.02 * limits[0], line_width=1, line_dash="solid", line_color="white")
    fig.add_hline(y=limits[1] + 0.02 * limits[1], line_width=1, line_dash="solid", line_color="white")
    #fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)