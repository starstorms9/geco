#%% Imports
import streamlit as st
header = st.title("Starting up...")
import numpy as np
import pandas as pd
import os
import sys
import time
import re
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64
from scipy.stats import pearsonr
import natsort as ns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from pathlib import Path
import subprocess

import streamlit.ReportThread as ReportThread
from streamlit.server.Server import Server

#%% Check for GPU and try to import TSNECuda
gpu_avail = False
try :
    from tsnecuda import TSNE as TSNECuda
    gpu_avail = True
except :
    print('TSNE Cuda could not be imported, not using GPU.')

#%% Helper Methods
nat_sort = lambda l : ns.natsorted(l)
# nat_sort = lambda l : sorted(l,key=lambda x:int(re.sub("\D","",x) or 0))
removeinvalidchars = lambda s : re.sub('[^a-zA-Z0-9\n\._ ]', '', s).strip()

def setWideModeHack():
    max_width_str = f"max-width: 1200px;"
    st.markdown( f""" <style> .reportview-container .main .block-container{{ {max_width_str} }} </style> """, unsafe_allow_html=True)

def getSessionID():
    # Hack to get the session object from Streamlit.
    ctx = ReportThread.get_report_ctx()
    this_session = None    
    current_server = Server.get_current()
    session_infos = Server.get_current()._session_info_by_id.values()

    st.write(current_server.__dict__)
    st.write(session_infos)

    for session_info in session_infos:
        st.write(session_info.__dict__)
        s = session_info.session
        if (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue) :
            this_session = s
            

    if this_session is None: raise RuntimeError("Oh noes. Couldn't get your Streamlit Session object")
    return id(this_session)

datadir = Path(str(getSessionID()) + '_data')

def getTableDownloadLink(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv_file = df.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download full data link (add .csv extension after download) </a>'
    return href

@st.cache(allow_output_mutation=True)
def getDataPlot(file_name) :
    dfgene = pd.read_csv(file_name)
    index = 0
    for i, col in enumerate(dfgene.columns) :
        if col.startswith('avg_') :
            index = i
            break
    cols = list(dfgene.columns)
    sample_names = cols[1:index]
    all_types = [col.rsplit('_',1)[0] for col in sample_names]
    all_types = nat_sort(list(set(all_types)))
    avg_cols = [col for col in dfgene.columns if (col.startswith('avg_') and not col == 'avg_type')]
    dfgene[ ['avg_type'] + avg_cols ] = dfgene[ ['avg_type'] + avg_cols].round(decimals = 3)
    return dfgene, all_types, sample_names, avg_cols

def getDataRaw() :
    dfgene, glen = getFile(title='Gene expression data')

    if glen == 0 : return None, 0
    return dfgene, glen

@st.cache
def processRawData(dfgene) :
    warning_messages = []
    all_types, sample_names = set(), []
    pat_ends_in_number_with_underscore = re.compile(r'_\d+$')
    dfgene.rename(columns={ dfgene.columns[0]: "geneid" }, inplace = True)

    len1 = len(dfgene)
    dfgene.dropna(subset=['geneid'], inplace=True)
    len2 = len(dfgene)
    dfgene.drop_duplicates(subset=['geneid'], inplace=True)
    len3 = len(dfgene)
    dfgene.dropna(inplace=True)
    len4 = len(dfgene)

    if len1 != len2 :
        warning_messages.append('\nWARNING: Omitted {} entries with no ID.'.format(len1-len2))
    if len2 != len3 :
        warning_messages.append('\nWARNING: Removed {} duplicate IDs.'.format(len2-len3))
    if len3 != len4 :
        warning_messages.append('\nWARNING: Removed {} entries with blanks for some data.'.format(len3-len4))

    for col in dfgene.columns[1:] :
        if not bool(pat_ends_in_number_with_underscore.search(col)) :
            all_types = []
            break
        splits = col.rsplit('_', 1)
        if len(splits) > 1 :
            sample_names.append(col)
            all_types.add(splits[0])

    all_types = nat_sort(list(all_types))
    sample_names = nat_sort(sample_names)

    # The list comprehension below *should* work efficiently but Streamlit has a caching bug and can't hash this type of thing correctly so it has to be done in a ridiculously overcomplicated way...
    # typecols = [[col for col in list(dfgene.columns) if col.startswith(typ)] for typ in np.array(all_types)]
    typecols = [list() for i in range(len(all_types))]
    for col in dfgene.columns :
        for prefix in all_types :
            if col.rsplit('_', 1)[0] == prefix :
                typecols[all_types.index(prefix)].append(col)

    for sn in sample_names :
        dfgene = dfgene[pd.to_numeric(dfgene[sn], errors='coerce').notnull()]
    len5 = len(dfgene)
    if len4 != len5 :
        warning_messages.append('\nWARNING: Removed {} entries with non numeric data. Check input for #REF or #DIV errors.'.format(len4-len5))
    if len1 != len5 :
        warning_messages.append('\nTotal entries remaining after cleaning: {}'.format(len5))

    for i in range(len(typecols)) :
        dfgene['avg_'+all_types[i]] = (dfgene.loc[:,typecols[i]].astype(float).mean(axis=1)).round(decimals=6).astype(float)

    avg_cols = [col for col in dfgene.columns if col.startswith('avg_')]
    dfgene['type'] = dfgene[avg_cols].idxmax(axis=1).str.slice(start=4)
    dfgene['avg_type'] = dfgene.apply(lambda row : row['avg_' + row.type], axis=1)

    avg_cols = nat_sort(avg_cols)
    checkDeleteTempVects()
    return dfgene, all_types, sample_names, typecols, avg_cols, warning_messages

# This methods ensures every column header ends in a '_X' where X is the sample number
def dedupe(cols) :
    pat_ends_in_number_with_underscore = re.compile(r'_\d+$')
    cols = [removeinvalidchars(col) for col in cols]
    new_cols = []
    scnts = Counter()
    for col in cols :
        if cols.count(col) > 1 or not bool(pat_ends_in_number_with_underscore.search(col)):
            new_cols.append('{}_{}'.format(col, scnts[col.rsplit('_', 1)[0]] + 1))
        else :
            new_cols.append(col)
        scnts[col.rsplit('_', 1)[0]] += 1
    return new_cols

def getFile(title, sidebar=False, dedupe_headers=True) :
    file = st.file_uploader(title, type="csv") if not sidebar else st.sidebar.file_uploader(title, type="csv")
    df = None
    length = 0
    try :
        df = pd.read_csv(file)
        length = len(df)

        if dedupe_headers :
            headers = file.getvalue().partition('\n')[0].split(',')
            df.columns = [df.columns[0]] + dedupe(headers[1:])
    except :
        err_message = 'Error loading file or no file uploaded yet'
        _ = st.sidebar.text(err_message) if sidebar else st.write(err_message)
        checkDeleteTempVects()
    return df, length

def askColor(all_types) :
    extra_color_indices = ['Assigned type', 'Average expression of assigned type', 'Enrichment in type (select)']
    color_indices = ['Expression of {}'.format(typ) for typ in all_types] + extra_color_indices
    color_nums = ['avg_'+typ for typ in all_types] + ['type', 'avg_type', 'norm_control']
    chosen_color = color_nums[ color_indices.index(st.sidebar.selectbox('Color Data', color_indices))]
    if chosen_color == 'norm_control' :
        control = st.sidebar.selectbox('Select type for enrichment:', all_types)
        return chosen_color, 'avg_' + control
    return chosen_color, None

def askMarkers(dfgene, dfmarkers) :
    markers_selected = st.sidebar.selectbox('Select curated marker genes', ['none'] + list(dfmarkers.columns))
    markers = []
    if not markers_selected == 'none' :
        markers = list(dfmarkers[markers_selected].dropna().values)
        markers_found = dfgene[dfgene.geneid.isin(markers)]
        return markers_found
    return []

def askGids(dfgene) :
    gids_input = st.sidebar.text_area("Gene IDs (comma or space separated):")
    gids_inspect = [gid.strip().replace("'", "") for gid in re.split(r'[;,\s]\s*', gids_input) if len(gid) > 1]
    gids_found = dfgene[dfgene.geneid.isin(gids_inspect)]
    return gids_found

def askColorScale(chosen_color) :
    type_color = chosen_color == 'type'
    seq_color_scale = st.sidebar.checkbox('Use continuous color scale?', value=False) if type_color else True
    reverse_color_scale = st.sidebar.checkbox('Reverse color scale?', value=type_color)

    if not seq_color_scale :
        color_scale_options = [cso for cso in dir(px.colors.qualitative) if (not cso.startswith('_')) and (not cso.endswith('_r')) and (not cso=='swatches') ]
        color_scale = st.sidebar.selectbox('Discrete Color scale', color_scale_options, index=color_scale_options.index('G10'))
    else :
        color_scale_options = [cso for cso in dir(px.colors.sequential) if (not cso.startswith('_')) and (not cso.endswith('_r')) and (not cso=='swatches') ]
        color_scale = st.sidebar.selectbox('Sequential Color scale', color_scale_options, index=color_scale_options.index('Blackbody' if not type_color else 'Blues'))

    if reverse_color_scale :
        color_scale += '_r'
    return color_scale

def calcPlotLimits(dfgene, padding = 1.05) :
    xmin, xmax = dfgene.red_x.min(), dfgene.red_x.max()
    ymin, ymax = dfgene.red_y.min(), dfgene.red_y.max()
    xlims, ylims = [xmin*padding, xmax*padding], [ymin*padding, ymax*padding]
    return xlims, ylims

def plotReduced(dfgene, all_types, color_scale, chosen_color, gids_found, markers_found, xlims, ylims, log_color_scale) :
    if len(dfgene) == 0 :
        st.error('No points left after filters.')
        return

    color_log = dfgene[chosen_color]
    if log_color_scale :
        color_log = (np.log10(color_log + 1))
    if chosen_color != 'type' :
        color_log = color_log.round(decimals=3)
    cmin, cmax = color_log.min(), color_log.max()
    if cmin == cmax :
        cmax += .001

    if cmin == float('-inf') or cmax == float('inf') :
        st.error('Error taking log of color, check for negative values in input')
        return

    disc_color_scale = color_scale in dir(px.colors.qualitative)
    is_type_color = chosen_color == 'type'
    color_disc_seq = getattr(px.colors.qualitative if disc_color_scale else px.colors.sequential, color_scale)

    fig = px.scatter(dfgene, x="red_x", y="red_y",
                     color=color_log,
                     range_x = xlims, range_y = ylims,
                     width=1200, height=800,
                     hover_name='geneid',
                     hover_data= nat_sort([col for col in dfgene.columns if col.startswith('avg_')]),
                     category_orders={"type": all_types},
                     color_continuous_scale = color_scale,
                     color_discrete_sequence = color_disc_seq )
    fig.update_xaxes(title_text='x')
    fig.update_yaxes(title_text='y')
    fig.update_traces(marker= dict(size = 6 if len(dfgene) > 1000 else 12 , opacity=0.9, line=dict(width=0.0)))

    if not is_type_color :
        if log_color_scale :
            ticks = list(range(int(cmin-1),int(cmax+1)))
            tick_txt = ['{:,}'.format(10**tick) for tick in ticks]
        else :
            ticks = list(range(int(cmin),int(cmax+1), int(np.ceil((cmax-cmin)/10)) ))
            tick_txt = ['{:,}'.format(tick) for tick in ticks]

        fig.update_layout(coloraxis_colorbar=dict( title="Expression", tickvals=ticks, ticktext=tick_txt))

    if len(gids_found) > 0:
        fig.add_scatter(name='Genes', text=gids_found.geneid.values, mode='markers', x=gids_found.red_x.values, y=gids_found.red_y.values, line=dict(width=5), marker=dict(size=20, opacity=1.0, line=dict(width=3)) )

    if len(markers_found) > 0 :
        fig.add_scatter(name='Markers', text=markers_found.geneid.values, mode='markers', x=markers_found.red_x.values, y=markers_found.red_y.values, line=dict(width=1), marker=dict(size=20, opacity=1.0, line=dict(width=3)) )

    fig.update_layout(legend=dict(x=0, y=0))
    st.plotly_chart(fig, use_container_width=False)

def plotExpressionInfo(gids_found, sample_names, all_types, avg_cols) :
    dfsub = gids_found.loc[:, list(sample_names)]
    dfsub.index = gids_found.geneid
    dfsub = dfsub.T
    types = [sn.rsplit('_', 1)[0] for sn in sample_names]

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(10, 4)
    axes.set_title("Expression of Gene per Type", fontsize=20)
    geneidtoplot = st.sidebar.selectbox('Gene to bar plot', nat_sort(gids_found.geneid.values))
    sns.barplot(x=types, y=geneidtoplot, data=dfsub, capsize=.05, errwidth=1.0)
    sns.swarmplot(x=types, y=geneidtoplot, data=dfsub, color="0", alpha=.75)
    st.pyplot()

    if len(gids_found) > 1 :
        st.header('Correlation Clustermap for Selected Genes')
        ppval = lambda x,y : pearsonr(x,y)[1]
        dfcorr = dfsub.corr(method='pearson')
        dfcorrp = dfsub.corr(method=ppval)

        pval_min = 0.01
        labels = pd.DataFrame().reindex_like(dfcorrp)
        labels[dfcorrp >= pval_min] = ''
        labels[dfcorrp < pval_min] = '*'
        np.fill_diagonal(labels.values, '*')

        g = sns.clustermap(dfcorr, center=0, annot=labels.values, fmt='', linewidths=0.01, cbar_kws={'label': 'Correlation'}, cmap='BrBG')
        ax = g.ax_heatmap
        ax.set_xlabel('')
        ax.set_ylabel('')
        st.pyplot()

    if len(gids_found) > 1 :
        st.header('Expression Heatmap for Selected Genes')
        fig, axes = plt.subplots(1, 1)
        normalize = st.sidebar.checkbox('Normalize heatmap?', value=True)
        dfheatmap = dfsub if not normalize else dfsub / dfsub.max()

        g = sns.clustermap(dfheatmap.T, cmap="Blues", col_cluster=False, cbar_kws={'label': 'Expression'},
                            figsize=(10, 3 + 0.25 * len(gids_found)), linewidths=0.2)

        ax = g.ax_heatmap
        ax.set_ylabel('')
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        st.pyplot()

def selectGenes(dfgene) :
    get_all = st.sidebar.button('Get all genes')
    limits = ['Reduced X min', 'Reduced X max', 'Reduced Y min', 'Reduced Y max']
    lims_sel = [st.sidebar.number_input(lim, value=0) for lim in limits]
    lims_all = [-9999, 9999, -9999, 9999]
    lims = lims_all if get_all else lims_sel

    selected_genes = dfgene[(dfgene.red_x > lims[0]) & (dfgene.red_x < lims[1]) &
                            (dfgene.red_y > lims[2]) & (dfgene.red_y < lims[3])]

    if len(selected_genes) > 0 :
        st.header('List of selected genes in window')
        st.markdown(getTableDownloadLink(selected_genes), unsafe_allow_html=True)
        st.text(',\n'.join(selected_genes.geneid.values))
        
def checkDeleteTempVects() :
    try : os.remove(datadir / 'temp_dfreduce.csv')
    except : pass

def deleteOldSessionData() :
    command_delOldData = 'find *_data* -maxdepth 3 -name \'*dfreduce*.csv\' -type f -mtime +5 -exec rm {} \\;'
    command_rmEmptyDir = 'find *_data* -empty -type d -delete'
    subprocess.Popen(command_delOldData, shell=True)
    subprocess.Popen(command_rmEmptyDir, shell=True)

#%% Main Methods
def plotData() :
    header.title('Plotting Gene Data')
    setWideModeHack()
    checkDeleteTempVects()
    checkMakeDataDir()

    csv_files = [os.path.join(datadir, fp).replace('\\','/') for fp in os.listdir(os.path.join(os.getcwd(), datadir)) if fp.startswith('dfreduce_') and fp.endswith('.csv')]
    if len(csv_files) == 0 :
        st.write('No files found. Generate files in the Generate reduced data mode.')
        return
    
    st.sidebar.header('Load Data')
    file_names_nice = nat_sort([fp.replace( str(datadir / 'dfreduce_'),'') for fp in csv_files])
    file_name = str(datadir / 'dfreduce_') + st.sidebar.selectbox('Select data file', file_names_nice)
    if st.sidebar.button('Delete this dataset') :
        os.remove(file_name)
        st.success('File \'{}\' removed, please select another file.'.format(file_name.replace( str(datadir / 'dfreduce_'), '')))
        return

    # Load Data
    dfgene, all_types, sample_names, avg_cols = getDataPlot(file_name)
    dfplot = dfgene.copy(deep=True)
    
    # Get Inputs
    st.sidebar.header('Plot View Parameters')
    chosen_color, norm_control = askColor(all_types)
    log_color_scale = False if chosen_color == 'type' else st.sidebar.checkbox('Log color scale?', value=True)
    color_scale = askColorScale(chosen_color)

    if chosen_color == 'norm_control' and norm_control :
        avg_cols_but_one = avg_cols.copy()
        avg_cols_but_one.remove(norm_control)
        dfplot['norm_control'] = dfplot[norm_control].div(dfplot[avg_cols_but_one].mean(axis=1)).round(3)
        dfplot['norm_control'] = dfplot['norm_control'].replace(np.inf, -np.inf)
        dfplot['norm_control'] = dfplot['norm_control'].replace(-np.inf, dfplot['norm_control'].max())

    avg_typ_min, avg_typ_max = min(0.0,float(dfplot.avg_type.min())), float(dfplot.avg_type.max())
    min_expression_typ = st.sidebar.number_input('Min expression of assigned type to show:', min_value = avg_typ_min, value = avg_typ_min, max_value = avg_typ_max, step=0.1)

    if not (chosen_color == 'type' or chosen_color == 'avg_type') :
        avg_sel_min, avg_sel_max = min(0.0,float(dfplot[chosen_color].min())),  float(dfplot[chosen_color].max())
        sel_min_title = 'Fold change of selected type over average of other types:' if chosen_color == 'norm_control' else 'Min expression of {} to show:'.format(chosen_color[4:])
        min_expression_sel = None if (chosen_color == 'type' or chosen_color == 'avg_type') else st.sidebar.number_input(sel_min_title, min_value = avg_sel_min, value = avg_sel_min, max_value=avg_sel_max, step=0.01 if chosen_color == 'norm_control' else 0.1)

    st.sidebar.header('Gene Markers to Show')
    markers_found = []
    dfmarkers, mlen = getFile(title='Gene gene marker data (optional)', sidebar=True, dedupe_headers=False)
    if mlen > 0 : markers_found = askMarkers(dfplot, dfmarkers)
    gids_found = askGids(dfplot)

    # Plot all the things
    xlims, ylims = calcPlotLimits(dfplot)
    dfplot = dfplot[dfplot.avg_type >= min_expression_typ]
    if not (chosen_color == 'type' or chosen_color == 'avg_type') :
        dfplot = dfplot[dfplot[chosen_color] >= min_expression_sel]

    # Main plot
    plotReduced(dfplot, all_types, color_scale, chosen_color, gids_found, markers_found, xlims, ylims, log_color_scale)

    # Plot the expression of the genes per type
    if len(gids_found) > 0:
        plotExpressionInfo(gids_found, sample_names, all_types, avg_cols)

    # List out the marker genes found
    if len(markers_found) > 0 and st.sidebar.checkbox('Show marker gene list?', value=False) :
        st.header('List of curated marker genes')
        st.write(',\n'.join(markers_found.geneid.values))

    # Give simple download interface for gene data
    st.sidebar.header('Filtered Gene Download')
    selectGenes(dfplot)

def readMe() :
    header.title('GECO - README')
    st.markdown("""           
        Welcome to GECO (Gene Expression Clustering Optimization), the straightforward, user friendly [Streamlit] app to visualize and investigate data patterns with non-linear reduced dimensionality plots.
        
        Although developed for bulk RNA-seq data, it will analyze any .csv data matrix with sample names (columns) and type (rows) [type = genes, protein, any other unique ID]. The output is an interactive and customizable t-SNE/UMAP. This visualization is intended to supplement more traditional statistical differential analysis pipelines (for example DESeq2 for bulk RNA-seq) and confirm or also reveal new patterns. 
        
        ### File Upload
        *** (required) Data Matrix. ***
        * Must be supplied as a .csv file.
        * The first column should contain the unique IDs for this dataset (genes, isoforms, protein, or any other identifier) which will be renamed ‘geneid’ in the program. Each unique ID should have ‘expression’ data listed in each row that corresponds to each sample. 
        * Sample names must be listed at the top of each columns, with biological replicates being indicated by ‘_#’ following sample name. Biological replicates are averaged during the analysis and the number of biological replicates does not need to match. For example, samples are named ‘WT’, ‘KO’, and ‘OE’ with three biological replicates each, the column names should be as shown in the example below.
        * If no ‘_#’ columns are found for a given sample name but there are duplicate column names, they will automatically have sample numbers appended.
        No index column (1,2,3,4) or other columns with additional information. 
        'NA' entries will be interpreted as unknowns and those entire rows will be removed
        * Also any of: ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', 'NA', '', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']
        * If you want to include these rows with non numeric values you must impute the NA values manually and replace them. This can be done by 0’ing them out, averaging with other samples, etc.
        
        
        *** (optional) Curated Markers List. ***
        * Must be supplied as a .csv file.
        * Each column should start with a descriptive title (will appear in a drop-down list). Below the title will be a list of unique IDs (that overlap with the provided data matrix). 
        * Multiple curated lists can be provided by listing them next to each other, one per column. Do not skip columns, and do not use different excel sheet tabs. 
        * This is capital sensitive, so make sure your capitalization is consistent. A minimal example is provided below.  
            
        
        ### Usage Instructions 
        **Generate reduced dimensionality data**
        
        1. Start on the ‘Generate reduced data’ mode (May need to hit the ‘>’ in top left to reveal the sidebar menu with Select Mode option.)
        
        2. Upload data matrix in the format described above. Review the inferred sample information – if it is not correct, update the .csv file accordingly.
        
        3. Set-up Reduction Run Parameters
            - See info [here][1] for setting t-SNE parameters
            - See info [here][2] for setting UMAP parameters
            - *Three optional boxes can be checked:*
                        
                1. *Remove entries with all zeros*
                
                2. *Normalize per gene (row)* – this divides each entry of a given row by the sum of the entire row and thus allows for investigation of trends across samples independent of overall expression level.
                
                3. *Normalize to control* – This normalizes each row to the control (a drop-down menu will allow you to select the control). This allows for investigation of trends relating to the fold change compared to a control. 
        
        4. Run the reduction. This could take a bit of time depending on the size of the data. You will see ‘running’ in the middle of the app.
        
        5. Some preliminary data and manipulation are available. Enter a filename and save the data.
            - *Note that using the same filename as one that already exists will overwrite the file.*
        
        **Visualize the data**
        
        6. Switch to ‘Plot reduced data’ mode 
        
        7. Select the desired saved dataset from the drop-down list under ‘Load Data’.
        
        8. Each dot/data point on the t-SNE/UMAP corresponds to one gene/protein/other unique ID depending on the input data. Using the ‘Color Data’ pull down menu the coloring of these data points can be altered in several ways:
            - Sample name = Average expression based on each sample type – one at a time. 
            - Type = each data point is assigned the color of the sample type that has the highest expression for that gene/protein/other unique ID.
            - Type average = Color scale set to range of expression and each data point is colored for the average of the sample type that has the highest expression.
            - Normalize to control = Color scale set to range of fold change of control over maximum sample type expression (color corresponding to highest expression will mark data points that are highest in the selected control relative to other sample types). 
        
        9. Some additional color/display options:
            - Log scale [only for some color data options] – log transforms the scale of colors and is useful if there are some prominent outliers over shadowing other data points. 
            - Continuous color scale [only for some color data options] - uses a continuous color gradient across discrete types to show transitions from one sample to the next. Particularly useful for time course or drug treatment when sample types are related. 
            - Reverse color scale switches the order of the colors.
            - Sequential/Discrete color scales changes the overall colors used. 
        
        10. Additional filtering steps:
            - Minimum assigned type expression removes data points with low expression based on the number in the filter. Filters based on the ‘assigned type’ which is the sample with the maximum expression. 
            - Min expression of type [only for average of sample type color option] removes data points below a specified threshold for the specific sample type selected to colorize the data. 
            - Fold change filter [only for normalize to control option] removes data points below a specified cut-off between designated control and non-control. 
        
        11. ‘Gene Markers to Show’ will highlight the specified genes/proteins/unique IDs.
            - Curated markers list can be uploaded as a .csv file (genes in a column with descriptive header – files with multiple columns are accepted; see description in file upload format). The descriptive header will be used to select which marker gene list should be displayed. These will be highlighted on the plot with an ID dot which is a large dot with a black outline. 
            - Gene IDs input list which will be highlighted with an ID dot that is a large dot with a black outline.  
            - This is case sensitive and must match the given dataset – if a gene/protein/unique ID is not found it will not be displayed (no warning/error message), but other matching IDs from the list will be displayed.
            - The ID dot will be a large circle appearing on the plot behind the data point (depending on your color scale you may need to adjust to see it if it is in a dense area). A key ‘Genes’ and/or ‘Markers’ will appear on the bottom left corner. By clicking it you will hide the circles while preserving the gene list. 
        12. Once at least one gene has been correctly entered in the Gene ID box a bar graph will appear below the t-SNE/UMAP and plot the gene across the samples. If more than one gene has been entered the one displayed in the bar graph can be changed using the drop down ‘Gene to bar plot’ menu. 
        
        13. Once at least two genes have been correctly entered in the Gene ID box two additional displays will appear:
            - A clustermap showing the correlations of the selected genes will appear below the bar graph. This will display all of the genes and calculate a correlation coefficient showing an asterisk for a correlation or anti-correlation that is significant.  
            - An expression heatmap of the specified genes across all samples. This is by default normalized by gene but the box that appears can be un-checked to disable this.
        
        14. Under the ‘Filtered Gene Download’ there is a ‘Get all genes’ button that will print a list of all of the genes/proteins/unique IDs in the dataset. To print a specified cluster of genes type in the window of x and y coordinates and all genes from that specific range will be printed (with any plotting filters applied). Once the gene list is printed to the screen it can be downloaded (add the correct .csv extension to the end of the file name before opening).
        
        15. The gene displayed can be adjusted by filtering as specified earlier or by zooming in/out on the plot. Zooming in can be performed by highlighting an area or hovering over the plot until a set of buttons appears in the top right corner (including +/- buttons). Double clicking on the plot returns to the default view. Hovering over a data point will display the gene and information on the expression in samples and the current color scale information. 
        
        ### Troubleshooting / FAQ 
        
        *File uploader utility says ‘files are not allowed’.*
        - Check that the file ends in a .csv and is a simple comma separated table.
        
        Odd persistent issues with the app.*
        - You can soft reboot the app by hitting ‘c’ and ‘clear cache’ and then hit ‘r’ to reload. 
        
        *TSNE takes a long time to run, how can I make it faster?*
        - To reduce the runtime of the t-SNE algorithm a GPU can be used. This requires a CUDA enabled graphics card (most Nvidia GPU’s), a Linux based system, and a more complex installation. However, using a GPU will reduce the t-SNE runtime down to only a few seconds for even very large datasets.
        
        *What correlation metric is used for the clustermap?*
        - A Pearson r correlation test is used. More info [here][3].
        
        *What are some suggested color scales?*
        - Blackbody
        - Electric
        - Jet
        - Thermal         
        
        [Streamlit]: <https://www.streamlit.io/>
        [1]: <https://distill.pub/2016/misread-tsne/>
        [2]: <https://umap-learn.readthedocs.io/en/latest/parameters.html>
        [3]: <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>
        
            """, unsafe_allow_html=True )
    

def checkMakeDataDir() :
    if os.path.exists(datadir) :
        return
    os.mkdir(datadir)

def umapReduce(npall, n_neighbors=15, min_dist=0.1, metric='euclidean') :
    print('Running UMAP...')
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    vects = reducer.fit_transform(npall)
    return vects

def tsneReduce(npall, pca_components=0, perp=40, learning_rate=200, n_iter=1000, early_exaggeration=12) :
    if pca_components > 0 :
        pca = PCA(n_components=pca_components)
        princcomps = pca.fit_transform(npall)
    else :
        princcomps = npall

    if gpu_avail :
        try :
            tsne_vects_out = TSNECuda(n_components=2, early_exaggeration=early_exaggeration, perplexity=perp, learning_rate=learning_rate).fit_transform(princcomps)
            return tsne_vects_out
        except Exception as e:
            st.warning('Error using GPU, defaulting to CPU TSNE implemenation. Error message:\n\n' + str(e))

    tsne = TSNE(n_components=2, n_iter=n_iter, verbose=3, perplexity=perp, method='barnes_hut', angle=0.5, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_jobs=-1)
    out_vects = tsne.fit_transform(princcomps)
    return out_vects

def genData() :
    header.title('Generate reduced dimensionality data')
    dfgene, glen = getDataRaw()
    if glen == 0 : return
    try :
        dfgene, all_types, sample_names, typecols, avg_cols, warnings = processRawData(dfgene)
        if len(warnings) > 0 :
            st.warning('\n\n'.join(warnings))
    except Exception as e:
        st.error('Error processing data, please check file requirements in read me and reupload. Error:\n\n' + str(e))
        return

    dfprint = pd.DataFrame(typecols).T.fillna('')
    dfprint.columns = all_types
    dfprint.index = ['Sample #{}'.format(i) for i in range(1, len(dfprint)+1)]
    st.info('Review inferred samples (rows) and types (columns) table below:')
    st.dataframe(dfprint)

    st.sidebar.header('Reduction Run Parameters')
    ralgo = st.sidebar.selectbox('Reduction algorightm:', ['UMAP', 'TSNE'])
    useUmap = ralgo == 'UMAP'

    param_guide_links = ['https://distill.pub/2016/misread-tsne/', 'https://umap-learn.readthedocs.io/en/latest/parameters.html']
    st.sidebar.markdown('<a href="{}">Guide on setting {} parameters</a>'.format(param_guide_links[useUmap], ralgo), unsafe_allow_html=True)
    remove_zeros = st.sidebar.checkbox('Remove entries with all zeros?', value=True)
    
    if len(avg_cols) > 2 :
        norm_per_row = st.sidebar.checkbox('Normalize per gene (row)?', value=True) 
        norm_control = st.sidebar.checkbox('Normalize to control?', value=False)
    else :
        norm_per_row, norm_control = False, False
        st.sidebar.text('Cannot normalize per row or control with only 2 types')    
    
    if norm_control :
        control = st.sidebar.selectbox('Select control:', all_types)

    if not useUmap :
        pca_comp = st.sidebar.number_input('PCA components (0 to run only TSNE)', value=0, min_value=0, max_value=len(all_types)-1, step=1)
        perp = st.sidebar.number_input('TSNE Perplexity', value=50, min_value= 40 if gpu_avail else 2, max_value=10000, step=10)
        learning_rate = st.sidebar.number_input('TSNE Learning Rate', value=200, min_value=50, max_value=10000, step=25)
        exagg = st.sidebar.number_input('TSNE Early Exaggeration', value=12, min_value=0, max_value=10000, step=25)
        if not gpu_avail : max_iterations = st.sidebar.number_input('TSNE Max Iterations', value=1000, min_value=500, max_value=2000, step=100)
        else : max_iterations = 1000
    else :
        n_neighbors = st.sidebar.number_input('UMAP Number of neighbors', value=15, min_value=2, max_value=10000, step=10)
        min_dist = st.sidebar.number_input('UMAP Minimum distance', value=0.1, min_value=0.0, max_value=1.0, step=0.1)
        umap_metrics = ['euclidean','manhattan','chebyshev','minkowski','canberra','braycurtis','haversine','mahalanobis','wminkowski','seuclidean','cosine','correlation']
        umap_metric = st.sidebar.selectbox('UMAP Distance Metric:', umap_metrics)

    if st.sidebar.button('Run {} reduction'.format(ralgo)) :
        status = st.header('Running {} reduction'.format(ralgo))
        dfreduce = dfgene.copy(deep=True)

        if remove_zeros :
            dfreduce = dfreduce.loc[(dfreduce[avg_cols]!=0).any(axis=1)]
        if norm_control or norm_per_row :
            dfreduce[avg_cols] = dfreduce[avg_cols] + sys.float_info.epsilon
        if norm_control :
            dfreduce[avg_cols] = dfreduce[avg_cols].div(dfreduce['avg_'+control], axis=0)
        if norm_per_row :
            dfreduce[avg_cols] = dfreduce[avg_cols].div(dfreduce[avg_cols].sum(axis=1), axis=0)
        if norm_control or norm_per_row :
            dfreduce[avg_cols] = dfreduce[avg_cols].round(decimals=4)

        if (dfreduce[avg_cols].isna().sum().sum() > 0) :
            st.write('!Warning! Some NA values found in data, removed all entries with NAs, see below:', dfreduce[avg_cols].isna().sum())
            dfreduce = dfreduce.dropna()

        data_vects_in = dfreduce[avg_cols].values + sys.float_info.epsilon

        start = time.time()
        if not useUmap :
            lvects = tsneReduce(data_vects_in, pca_components=pca_comp, perp=perp, learning_rate=learning_rate, n_iter=max_iterations, early_exaggeration=exagg)
        else :
            lvects = umapReduce(data_vects_in, n_neighbors, min_dist, umap_metric)
        st.write('Reduction took {:0.3f} seconds'.format((time.time()-start) * 1))

        dfreduce['red_x'] = lvects[:,0]
        dfreduce['red_y'] = lvects[:,1]
        checkMakeDataDir()
        dfreduce.round(decimals=4).to_csv(datadir / 'temp_dfreduce.csv', index=False)
    elif not os.path.exists(datadir / 'temp_dfreduce.csv') :
        return
    else :
        status = st.header('Loading previous vectors')
        dfreduce = pd.read_csv(datadir / 'temp_dfreduce.csv')

    st.sidebar.header('Plot Quick View Options')
    form_func = lambda typ : 'Expression of {}'.format(typ) if typ != 'Type' else typ
    chosen_color = st.sidebar.selectbox('Color data', ['Type'] + all_types, format_func=form_func)
    hue = 'type' if chosen_color == 'Type' else 'avg_' + chosen_color

    if chosen_color == 'Type' :
        ax = sns.scatterplot(data=dfreduce, x='red_x', y='red_y', s=5, linewidth=0.01, hue=hue)
        ax.set(xticklabels=[], yticklabels=[], xlabel='{}_x'.format(ralgo), ylabel='{}_y'.format(ralgo))
        plt.subplots_adjust(top=0.98, left=0.05, right=1, bottom=0.1, hspace=0.0)
    else :
        fig, ax = plt.subplots(1)
        plt.scatter(x=dfreduce.red_x.values, y=dfreduce.red_y.values, s=5, linewidth=0.01, c=dfreduce[hue].values, norm=matplotlib.colors.LogNorm())
        plt.colorbar(label='Expression Level')
        plt.subplots_adjust(top=0.98, left=0.05, right=1, bottom=0.1, hspace=0.0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    status.header('Data plot quick view:')
    st.pyplot()
    st.write('Total number of points: ', len(dfreduce))

    st.sidebar.header('Reduced Data Save')

    if not useUmap :
        suggested_fn = '{}p'.format(perp)
        suggested_fn += '_No0' if remove_zeros else ''
        suggested_fn += '_NR' if norm_per_row else ''
        suggested_fn += '_NC-'+control if norm_control else ''
    else :
        suggested_fn = '{}n{}m'.format(n_neighbors, int(100*np.round(min_dist, 2)))
        suggested_fn += '_met-{}'.format(umap_metric) if not umap_metric == 'euclidean' else ''
        suggested_fn += '_No0' if remove_zeros else ''
        suggested_fn += '_NR' if norm_per_row else ''
        suggested_fn += '_NC-'+control if norm_control else ''

    file_name = removeinvalidchars(st.sidebar.text_input('Data file name:', value=suggested_fn, max_chars=150))
    if len(file_name) > 0 and st.sidebar.button('Save data file') :
        dfsave = dfgene.copy()
        dfsave = pd.merge(dfsave, dfreduce[['geneid', 'red_x', 'red_y']], on='geneid', how='right')
        dfsave = dfsave.round(decimals=3)
        checkMakeDataDir()
        dfsave.to_csv( str(datadir / 'dfreduce_') + file_name + '.csv', index=False)
        st.success('File \'{}\' saved!'.format(file_name))

#%% Main program execution
modeOptions = ['Read Me', 'Generate reduced data', 'Plot reduced data']
st.sidebar.image('GECO_logo.jpg', use_column_width=True)
st.sidebar.header('Select Mode:')
mode = st.sidebar.radio("", modeOptions, index=0)
tabMethods = [readMe, genData, plotData]
tabMethods[modeOptions.index(mode)]()
deleteOldSessionData()
