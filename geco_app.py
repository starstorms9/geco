###############
### Imports ###
###############

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

# Check for GPU and try to import TSNECuda if possible. Otherwise just use sklearn
gpu_avail = False
try :
    from tsnecuda import TSNE as TSNECuda
    gpu_avail = True
except :
    print('TSNE Cuda could not be imported, not using GPU.')

######################
### Helper Methods ###
######################

nat_sort = lambda l : ns.natsorted(l)
removeinvalidchars = lambda s : re.sub('[^a-zA-Z0-9\n\._ ]', '', s).strip()

def setWideModeHack():
    '''
    Streamlit hack to set wide mode programmatically so that it doesn't have to
    be selected every time.

    Returns
    -------
    None.

    '''
    max_width_str = f"max-width: 1200px;"
    st.markdown( f""" <style> .reportview-container .main .block-container{{ {max_width_str} }} </style> """, unsafe_allow_html=True)

def getSessionID():
    '''
    Streamlit hack to use report server debugging to get unique session IDs.
    These session IDs are used in order to separate data from different users
    and allows for easy sharing of plots using session IDs.

    Raises
    ------
    RuntimeError
        If the streamlit session object is not available.

    Returns
    -------
    sessionID
        The ID of the current streamlit user session.

    '''
    ctx = ReportThread.get_report_ctx()
    this_session = None    
    current_server = Server.get_current()
    session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue) :
            this_session = s
            
    if this_session is None: raise RuntimeError("Oh noes. Couldn't get your Streamlit Session object")
    return id(this_session)

def getTableDownloadLink(df):
    '''
    Streamlit hack to generate a link allowing the data in a given panda dataframe to be downloaded.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to be converted to download link.

    Returns
    -------
    href : string
        HTML string to insert into the output which will be shown as a URL link
        that will download the dataframe.

    '''
    csv_file = df.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download full data link (add .csv extension after download) </a>'
    return href

@st.cache(allow_output_mutation=True)
def getDataPlot(file_name) :
    '''
    Load data to plot. The output of this function is cached by streamlit
    in order to speed up subsequent calls.

    Parameters
    ----------
    file_name : string
        The file name of the data to plot.

    Returns
    -------
    dfgene : pandas DataFrame
        The full gene dataframe.
    all_types : list of strings
        All sample types that were found.
    sample_names : list of strings
        All valid sample names including sample numbers.
    avg_cols : list of strings
        The columns to average over.
    '''
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
    '''
    Passthrough function to ensure that a dataframe is read and if not then
    None is returned instead.

    Returns
    -------
    dfgene : pandas DataFrame
        The read dataframe or None if no dataframe was found.
    glen : 
        The length of the dataframe. 0 if no dataframe was found.
    '''
    st.subheader("Gene expression data upload:")
    st.write('''*Note that GECO does not perform any checks on the input data for
             statistical significance. If this is important for the analysis the 
             data must be preprocessed before uploading.*''')
    dfgene, glen = getFile(title='')

    if glen == 0 : return None, 0
    return dfgene, glen

def hash_df(df) :
    '''
    Custom hash function for data processing to ensure streamlit hashes the
    entire dataframe correctly. At a user level, this ensures that any changes
    to the uploaded data file are properly reflected downstream.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to hash.

    Returns
    -------
    hashed : int
        The hash of the entire dataframe.

    '''
    hashed = pd.util.hash_pandas_object(df).sum() + sum([hash(col) for col in df.columns])
    return hashed

@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame : hash_df})
def processRawData(dfgene) :
    '''
    Function to clean and organize and automatically label data for 
    subsequent post processing.
    The output of this function is hashed with the custom hash function to ensure
    that streamlit properly accounts for all of the data in the dataframe as
    well as the column headers.

    Parameters
    ----------
    dfgene : pandas DataFrame
        The dataframe to process.

    Returns
    -------
    dfgene : pandas DataFrame
        The processed dataframe.
    all_types : list of strings
        All sample types that were found.
    sample_names : list of strings
        All valid sample names including sample numbers.
    typecols : list of lists of strings
        All of the entries from sample_names but arranged such that the rows
        align with the all_types and the columns are each sample repeat.
    avg_cols : list of strings
        The columns to average over.
    warning_messages : string
        Warning messages that were generated while cleaning the data.

    '''    
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

def dedupe(cols) :
    '''
    This methods ensures every column header ends in a '_X' where X is the sample number

    Parameters
    ----------
    cols : list of strings
        The columns to check for duplicates.

    Returns
    -------
    new_cols : list of strings
        The processed columns.

    '''
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
    '''
    File uploader UI to get the a .csv file through the streamlit interface.

    Parameters
    ----------
    title : string
        The name of the file to be uploaded.
    sidebar : bool, optional
        Whether to show in the sidebar or the main area. The default is False.
    dedupe_headers : TYPE, optional
        Whether to deduplicate the headers. The default is True.

    Returns
    -------
    df : pandas DataFrame
        The uploaded dataframe.
    length : int
        The length of the uploaded dataframe. 0 if no data is uploaded.

    '''
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
    '''
    Streamlit UI for choosing how to colorize the main plot.

    Parameters
    ----------
    all_types : list of strings
        The possible types to color by.

    Returns
    -------
    chosen_color : string
        The chosen coloration option.
    norm_control
        Whether to normalize to control. Either avg_<chosen_control> or None to
        signify to not use this option.

    '''
    extra_color_indices = ['Assigned type', 'Average expression of assigned type', 'Enrichment in type (select)']
    color_indices = ['Expression of {}'.format(typ) for typ in all_types] + extra_color_indices
    color_nums = ['avg_'+typ for typ in all_types] + ['type', 'avg_type', 'norm_control']
    chosen_color = color_nums[ color_indices.index(st.sidebar.selectbox('Color Data', color_indices))]
    if chosen_color == 'norm_control' :
        control = st.sidebar.selectbox('Select type for enrichment:', all_types)
        return chosen_color, 'avg_' + control
    return chosen_color, None

def askMarkers(dfgene, dfmarkers) :
    '''
    Streamlit UI for processing a curated list of gene markers to show on the plot.

    Parameters
    ----------
    dfgene : pandas DataFrame
        Full dataframe containing all of the genes being analyzed.
    dfmarkers : pandas DataFrame
        Dataframe that was uploaded by the user containing the curated marker
        gene list to visualize.

    Returns
    -------
    markers_found
        Return any gene markers that were found. Returns empty list if no
        valid marker genes are found.

    '''
    markers_selected = st.sidebar.selectbox('Select curated marker genes', ['none'] + list(dfmarkers.columns))
    markers = []
    if not markers_selected == 'none' :
        markers = list(dfmarkers[markers_selected].dropna().values)
        markers_found = dfgene[dfgene.geneid.isin(markers)]
        return markers_found
    return []

def askGids(dfgene) :
    '''
    Streamlit UI for getting gene IDs (gids) from the user so that they can be
    visualized on the plot. Processes the gene IDs to go from a separated string
    to a list of valid gene entries in the full gene list.

    Parameters
    ----------
    dfgene : pandas DataFrame
        Full gene dataframe to check with to ensure the marker genes are valid.

    Returns
    -------
    gids_found : list of strings
        List of valid markers genes that were found.

    '''
    gids_input = st.sidebar.text_area("Gene IDs (comma or space separated):")
    gids_inspect = [gid.strip().replace("'", "") for gid in re.split(r'[;,\s]\s*', gids_input) if len(gid) > 1]
    gids_found = dfgene[dfgene.geneid.isin(gids_inspect)]
    return gids_found

def askColorScale(chosen_color) :
    '''
    Streamlit UI for getting the color scale. Depending on the data and 
    type of colorization chosen there are various color scale options.

    Parameters
    ----------
    chosen_color : string
        The colorization option chosen.

    Returns
    -------
    color_scale : string
        The plotly string defining the plotly internal color scale.

    '''
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
    '''
    Given the full gene list this utility methods outputs padded ranges for the plot.

    Parameters
    ----------
    dfgene : pandas DataFrame
        Full gene list from which to generate the bounding window limits.
    padding : float, optional
        The amount of padding to add. The default is 1.05 (5 % extra around the edges).

    Returns
    -------
    xlims : list of floats
        The x axis minimum and maximum.
    ylims : list of floats
        The y axis minimum and maximum.

    '''
    xmin, xmax = dfgene.red_x.min(), dfgene.red_x.max()
    ymin, ymax = dfgene.red_y.min(), dfgene.red_y.max()
    xlims, ylims = [xmin*padding, xmax*padding], [ymin*padding, ymax*padding]
    return xlims, ylims

def plotReduced(dfgene, all_types, color_scale, chosen_color, gids_found, markers_found, xlims, ylims, log_color_scale) :
    '''
    Main visualization method for showing the reduced data. After all necessary
    options have been defined this function generates and plots the data as a
    2D plotly chart.

    Parameters
    ----------
    dfgene : pandas DataFrame
        The full gene list to plot.
    all_types : list of strings
        All types of samples.
    color_scale : string
        The chosen color scale to use.
    chosen_color : string
        How to colorize the points.
    gids_found : list of strings
        Any valid gene markers specified by the user.
    markers_found : list of strings
        Any valid gene markers uploaded.
    xlims : list of floats
        The x axis limits.
    ylims : list of floats
        The y axis limits.
    log_color_scale : bool
        Whether to plot on a log scale or not.

    Returns
    -------
    None.

    '''
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
    '''
    Secondary plotting function to show the full expression info for a specific
    entry in the gene list.

    Parameters
    ----------
    gids_found : list of strings
        The gene IDs specified as options to plot.
    sample_names : list of strings
        The names of every sample uploaded including repetition number.
    all_types : list of strings
        All types of samples.
    avg_cols : list of strings
        List of columns to average over.

    Returns
    -------
    None.

    '''
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
    '''
    Utility method for getting genes from the reduced plot using window
    coordinates. Unfortunately getting the data directly from the Plotly
    plot was not feasible despite many attempts as it's a fundamental streamlit
    limitation.

    Parameters
    ----------
    dfgene : pandas DataFrame
        The full list of genes.

    Returns
    -------
    None.

    '''
    get_all = st.sidebar.button('Get all genes')
    limits = ['Reduced X min', 'Reduced X max', 'Reduced Y min', 'Reduced Y max']
    lims_sel = [st.sidebar.number_input(lim, value=0.0) for lim in limits]
    lims_all = [-9999.0, 9999.0, -9999.0, 9999.0]
    lims = lims_all if get_all else lims_sel

    selected_genes = dfgene[(dfgene.red_x > lims[0]) & (dfgene.red_x < lims[1]) &
                            (dfgene.red_y > lims[2]) & (dfgene.red_y < lims[3])]

    if len(selected_genes) > 0 :
        st.header('List of selected genes in window')
        st.markdown(getTableDownloadLink(selected_genes), unsafe_allow_html=True)
        st.text(',\n'.join(selected_genes.geneid.values))
        
def checkDeleteTempVects() :
    '''
    Delete vectors that were temporarily generated to quickly view.

    Returns
    -------
    None.

    '''
    try : os.remove(datadir / 'temp_dfreduce.csv')
    except : pass

def deleteOldSessionData() :
    '''
    Server utility function that runs command line functions on the server side
    to remove old and unused datasets after a while.

    Returns
    -------
    None.

    '''
    command_delOldData = 'find *_data* -maxdepth 3 -name \'*dfreduce*.csv\' -type f -mtime +15 -exec rm {} \\;'
    command_rmEmptyDir = 'find *_data* -empty -type d -delete'
    subprocess.Popen(command_delOldData, shell=True)
    subprocess.Popen(command_rmEmptyDir, shell=True)

####################
### Main Methods ###
####################

def plotData() :
    '''
    Main method for visualizing the reduced data that was previously generated.

    Returns
    -------
    None.

    '''
    header.title('Plotting Gene Data')
    setWideModeHack()
    checkDeleteTempVects()

    if not os.path.exists(datadir) :
        st.write('No files found. Generate files in the Generate reduced data mode.')
        return

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
    '''
    This simply shows the readme for geco from github as well as a quick
    intro video showing basic usage.

    Returns
    -------
    None.

    '''
    header.title('GECO - README')
    st.markdown("""
        Welcome to GECO (Gene Expression Clustering Optimization), the straightforward, user friendly [Streamlit app][Streamlit] to visualize and investigate data patterns with non-linear reduced dimensionality plots. Although developed for bulk RNA-seq data, GECO can be used to analyze any .csv data matrix with sample names (columns) and type (rows) [type = genes, protein, any other unique ID].
        
        The output is an interactable and customizable T-SNE/UMAP analysis and visualization tool. The visualization is intended to supplement more traditional statistical differential analysis pipelines (for example DESeq2 for bulk RNA-seq) and to confirm and/or reveal new patterns.

        If questions or issues arise please contact Amber Habowski at [Habowski@uci.edu](mailto:Habowski@uci.edu) or post an issue on the github issues page [here](https://github.com/starstorms9/geco/issues).
        
        [Streamlit]: <https://www.streamlit.io/>
        """, unsafe_allow_html=True )    
    
    st.header('GECO Demo in 3 minutes:')
    st.video('https://www.youtube.com/watch?v=wo8OW7eiJ5k')
    
    st.markdown("""
        ### Quick Guide to Getting Started
        1. Upload data file and verify that GECO has interpreted the sample names/bio-replicates.
        2. Select the reduction parameters to be used for the analysis.
        3. Click the 'Run UMAP/t-SNE reduction' button at the bottom of the parameters sidebar.
        4. Once a plot is generated, save it by clicking ‘Save data file’ at the bottom of the sidebar.
        5. Proceed to the ‘Plot reduced data’ mode to visualize the saved plot.
        
        ### File Upload 
        **_(required)_ The Data Matrix:**
        
        - Must be supplied as a .csv file.
        - The first column should contain the unique IDs for this dataset (genes, isoforms, protein, or any other identifier) which will be renamed 'geneid' in the program. Each unique ID should have 'expression' data listed in each row that corresponds to each sample.
        - Sample names must be listed at the top of each columns, with biological replicates being indicated by '\_#' following each sample name. Biological replicates are averaged during the analysis and the number of biological replicates does not need to match between samples. For example, for two samples ('Asample' and 'Bsample') with three biological replicates each the column names should be assigned as shown in the example below.
          - If no '\_#' columns are found for a given sample name, but there are duplicated column names, they will automatically have sample numbers appended.
        - The file should not have any index column (1,2,3,4) or other columns with additional information.
        - 'NA' entries will be interpreted as unknowns and those entire rows will be removed
          - Also any of: ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', 'NA', '', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']
          - If there is a need to include rows with non-numeric values, the NA values must be imputed manually and replaced. For example, this can be achieved by &quot;zero-ing&quot; non-numeric values, replacing NA values with average of samples, etc.
        - An example file for demo purposes is available [here](https://github.com/starstorms9/geco/blob/master/Supplement%232%20-%20ColonCryptCells.csv) (download the file from github and upload it to GECO.)
        
        **_(optional)_ Curated Markers List:**
        
        - Must be supplied as a .csv file.
        - Each column should start with a descriptive title (that will appear in a drop-down list). Below the title will be a list of unique IDs (that overlap with the provided data matrix). A minimal example is provided below.
        - Multiple curated lists can be provided by listing them next to each other, one per column. Do not skip columns, and do not use different excel sheet tabs.
        - GECO is case sensitive, so make sure capitalization of gene entry text is consistent with the provided Data Matrix.
        
        ### Usage Instructions
        **Session ID Information:**
        At the top of the sidebar is information on the current session ID and a box to input a previously saved session ID. This enables a previous session (including uploaded data and saved plots) to be reloaded for further investigation. This is also a great way to share saved analysis with collaborators if using the cloud-hosted version of GECO.
        
        **Generate reduced dimensionality data:**
        
        1. Start with the 'Generate reduced data' mode (click the '>' in the top left to reveal the sidebar menu with Select Mode option.)
        2. Upload data matrix in the format described above (&quot;File Upload&quot;). Review the inferred sample information from the upload – if the information is not correct, update the .csv file accordingly.
        3. Set Reduction Run Parameters:
           1. See info [here][1] for setting T-SNE parameters
           2. See info [here][2] for setting UMAP parameters
           3. Three optional boxes can be checked:
              - Remove entries with all zeros will simply ignore any entries that contain all zeros.
              - Normalize per gene (row) – this function divides each entry of a given row by the sum of the entire row and thus allows for investigation of trends across samples _independent_ of overall expression level.
              - Normalize to type – This function normalizes each row to a specified sample type (a drop-down menu will allow you to select a type for normalization) and allows for investigation of trends relating to the fold change compared to the selected type.
        4. Run the reduction. This could take a bit of time depending on the size of the data. Active analysis is visualized by a 'running' icon in the top right corner of the app.
        5. Preliminary data and manipulation are available. Enter a filename and save the data.
           - Note that using the same filename as one that already exists will overwrite the file.
        
        **Visualize the data:**
        
        1. Switch to 'Plot reduced data' mode
        2. Select the desired saved dataset from the drop-down list under 'Load Data'.
        3. Each dot/data point on the T-SNE/UMAP corresponds to one gene/protein/other unique ID depending on the input data. Using the 'Color Data' pull down menu the coloring of these data points can be altered in several ways:
           - Expression of sample_name/type = Average expression / value based on each sample type – one at a time.
           - Assigned type = Each data point is assigned the color of the sample type that has the highest expression/value for that gene/protein/other unique ID.
           - Average expression of assigned type = The color scale is set to a range of expression values and each data point is colored for the average of the sample type that has the highest expression.
           - Enrichment in type (select) = The color scale is set to a range of fold change of the selected sample (chosen in the dropdown) over maximum sample type expression. The color corresponding to the highest expression value will mark data points that are highest in the selected sample relative to other sample types.
        4. Additional color/display options:
           - Log Scale [only for some color data options] - transforms the scale of colors and is useful if there are prominent outliers overshadowing other data points.
           - The Continuous Color Scale option [only for some color data options] - uses a continuous color gradient across discrete types to show transitions from one sample to the next. This analysis mode is particularly useful for time course or drug treatment when sample types are related.
           - Reverse Color Scale switches the order of the colors.
           - Sequential/Discrete Color Scales changes the overall colors used.
        5. Additional filtering steps:
           - Min expression of assigned type to show = removes data points with low expression values based on the number in the filter. Filters are based on the 'assigned type' which is the sample with the maximum expression.
           - Min expression of _selected type_ to show [only available for &quot;Expression of _sample name/type&quot;_ color option] = removes data points below a specified threshold for the sample type selected to colorize the data.
           - Fold change of selected type over average of other types [only available for the &quot;Enrichment in type (select)&quot; color option] = removes data points below a specified cut-off between selected type and other types.
        6. 'Gene Markers to Show' will highlight the specified genes/proteins/unique IDs.
           - Curated marker lists can be uploaded as a .csv file (genes in a column with descriptive header – files with multiple columns are accepted; see description in file upload format). The descriptive header will be used to select which marker gene list should be displayed. These lists will be highlighted on the plot with an ID dot (which is a large dot with a black outline).
           - The Gene IDs input list will also be highlighted with an ID dot (the ID dot color will be different than the Curated list ID dots, if present).
           - This is case sensitive and must match the given dataset – if a gene/protein/unique ID is not found it will not be displayed (no warning/error message), but other matching IDs from the list will be displayed.
           - The ID dot will be a large circle appearing on the plot behind the data point. Depending on the color scale and density of data points, adjustments may need to be made to see the ID dot (filtering, zooming in, etc.). A key 'Genes' and/or 'Markers' will appear on the bottom left corner of the plot. By clicking on either of these terms, you will hide the circles while preserving the gene list.
        7. Once at least one gene has been correctly entered in the Gene ID box, a bar graph will appear below the T-SNE/UMAP and plot the gene across the samples. If more than one gene has been entered the one displayed in the bar graph can be changed using the drop down 'Gene to bar plot' menu.
        8. Once at least two genes have been correctly entered in the Gene ID box, two additional displays will appear:
           - A clustermap showing the correlations of the selected genes. This displays all of the genes and calculates a correlation coefficient (significance of correlation or anti-correlation is shown with an asterisk).
           - An expression heatmap of the specified genes across all samples. This is by default normalized by gene but the box that appears can be un-checked to disable this.
        9. Under the 'Filtered Gene Download' there is a 'Get all genes' button that will print a list of all of the genes/proteins/unique IDs in the dataset. To print a specified cluster of genes type in the window of x and y coordinates and all genes from that specific range will be printed (with any plotting filters applied). Once the gene list is printed to the screen it can be downloaded (add the correct .csv extension to the end of the file name before opening).
        10. The gene displayed can be adjusted by filtering as specified earlier or by zooming in/out on the plot. Zooming in can be performed by highlighting an area or hovering over the plot until a set of buttons appears in the top right corner (including +/- buttons). Double clicking on the plot returns to the default view. Hovering over a data point will display the gene and information on the expression in samples and the current color scale information.
        
        **Troubleshooting/FAQ**
        
        1. File uploader utility says 'files are not allowed'
           - Check that the file ends in a .csv and is a simple comma separated table.
        2. Odd persistent issues with the app
           - You can soft reboot the app by hitting 'c' and 'clear cache' and then hit 'r' to reload.
           - The normalization options for 'Normalize per gene (row)' and 'Normalize to type' are not appearing as options when generating reduced data.
           - These two options are only available when more than two sample types are included in the data matrix. When these normalizations are used on only two sample types it results in a useless analysis.
        3. T-SNE takes a long time to run when using GECO locally, how can I make it faster?
           - To reduce the runtime of the T-SNE algorithm a GPU can be used. This requires a CUDA enabled graphics card (most Nvidia GPU's), a Linux based system, and a more complex installation. However, using a GPU will reduce the T-SNE runtime down to only a few seconds for even very large datasets.
        4. Given the same settings and dataset, why do each generated T-SNE and UMAP not look the same?
           - Both UMAP and T-SNE are stochastic algorithms which mean there is a level of randomness to where datapoints initially fall into the plotted space. However, the relationships and trends between datapoints will be consistent.
        5. What is the best way to save GECO generated data?
           - It is recommended to save the current session ID number as the cloud hosted web browser page can get reloaded/cleared and it does eventually time-out after a long duration of use.
           - Plots can also be downloaded as a .png to a local folder using the camera capture button just above the top right side of the plot (hover the mouse over the plot if these buttons are not visible).
           - Once a gene list is generated a link will appear enabling the full list to be generated. Once this download occurs add the necessary '.csv' extension to the file name before opening.
        6. What correlation metric is used for the clustermap?
           - A Pearson R correlation test is used. More info [here][3].
        7. What are some suggested color scales?
           - Blackbody
           - Electric
           - Jet
           - Thermal

        [Streamlit]: <https://www.streamlit.io/>
        [GecoVid]: <https://youtu.be/wo8OW7eiJ5k>
        [1]: <https://distill.pub/2016/misread-tsne/>
        [2]: <https://pair-code.github.io/understanding-umap/>
        [3]: <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>

            """, unsafe_allow_html=True )    

def checkMakeDataDir() :
    '''
    Utility function to check whether the data directory exists and if not
    then make it.

    Returns
    -------
    None.

    '''
    if os.path.exists(datadir) :
        return
    os.mkdir(datadir)

def umapReduce(npall, n_neighbors=15, min_dist=0.1, metric='euclidean') :
    '''
    Use UMAP to reduce the data according to the input parameters.

    Parameters
    ----------
    npall : numpy array
        Numpy array of the input vectors that have been uploaded.
    n_neighbors : int, optional
        Number of nearest neighbors to consider when calculating distance metrics.
        The default is 15.
    min_dist : float, optional
        The minimum distance between points allowed in the output.
        The default is 0.1.
    metric : string, optional
        The distance metric to use. The default is 'euclidean'.

    Returns
    -------
    vects : numpy array
        The reduced vectors in the same order and rows as the input but with a
        dimension of only 2
    '''
    print('Running UMAP...')
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    vects = reducer.fit_transform(npall)
    return vects

def tsneReduce(npall, pca_components=0, perp=40, learning_rate=200, n_iter=1000, early_exaggeration=12) :
    '''
    Use TSNE to reduce the input data. Takes in arguments and passes them 
    directly to the TSNE algorithm.

    Parameters
    ----------
    npall : numpy array
        Numpy array of the input vectors that have been uploaded
    pca_components : int, optional
        If greater than 0 PCA is used before TSNE in order to reduce the runtime.
        The default is 0 (aka not using PCA)
    perp : float, optional
        Perplexity parameter for TSNE (basically how many nearest neighbors are 
        considered for computing distances). The default is 40.
    learning_rate : float, optional
        Learning rate per iteration. The default is 200.
    n_iter : int, optional
        Max number of iterations allowed. The default is 1000.
    early_exaggeration : float, optional
        Amount of early exaggeration to use in the TSNE algorithm. This
        increases the error in the first several iterations to facailitate
        convergence. The default is 12.

    Returns
    -------
    out_vects numpy array
        The reduced vectors in the same order and rows as the input but with a
        dimension of only 2
    '''
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
    '''
    This main function is the tab for generating the reduced data.
    It controls the flow of the user uploading the data, reducing the data,
    plotting it quickly to check, and saving the data to be fully plotted.

    Returns
    -------
    None.

    '''
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
    ralgo = st.sidebar.selectbox('Reduction algorithm:', ['UMAP', 'TSNE'])
    useUmap = ralgo == 'UMAP'

    param_guide_links = ['https://distill.pub/2016/misread-tsne/', 'https://pair-code.github.io/understanding-umap/']
    st.sidebar.markdown('<a href="{}">Guide on setting {} parameters</a>'.format(param_guide_links[useUmap], ralgo), unsafe_allow_html=True)
    remove_zeros = st.sidebar.checkbox('Remove entries with all zeros?', value=True)
    
    if len(avg_cols) > 2 :
        norm_per_row = st.sidebar.checkbox('Normalize per gene (row)?', value=True) 
        norm_control = st.sidebar.checkbox('Normalize to type?', value=False)
    else :
        norm_per_row, norm_control = False, False
        st.sidebar.text('Cannot normalize per row or control with only 2 types')    
    
    if norm_control :
        control = st.sidebar.selectbox('Select type for normalization:', all_types)
        
    hyperParamDescs = {'TSNE_PCA' : 'Specifies how many PCA components to reduce the input data to prior to running TSNE to speed up convergence. Enter 0 to disable PCA preprocessing.',
                       'TSNE_Perp' : 'Number of nearest neighbors considered when calculating clusters. Large values focus on capturing global structure at the cost of local details.',
                       'TSNE_LR' : 'Controls how fast TSNE creates clusters. Large values lead to faster convergence but too much can cause inaccuracies and divergence instead.',
                       'TSNE_EE' : 'Early exaggeration of the input points. Larger values force clusters to be more distinct by initially exaggerating their relative distances.',
                       'TSNE_MaxIt' : 'Maximum number of iterations for the TSNE algorithm. Larger values will increase accuracy but will take longer to compute.',
                       'UMAP_NN' : 'The number of nearest neighbors to consider for each input point. Large values focus on capturing global structure at the cost of local details.',
                       'UMAP_MinDist' : 'Minimum distance between output points. Controls how tightly points cluster together with low values leading to more tightly packed clusters.',
                       'UMAP_DistMetric' : 'Distance metric used to determine distance between points.'}

    if not useUmap :
        st.sidebar.markdown('''TSNE PCA Preprocess:''')
        pca_comp = st.sidebar.number_input(hyperParamDescs['TSNE_PCA'], value=0, min_value=0, max_value=len(all_types)-1, step=1)
        st.sidebar.markdown('''TSNE Perplexity:''')
        perp = st.sidebar.number_input(hyperParamDescs['TSNE_Perp'], value=50, min_value= 40 if gpu_avail else 2, max_value=10000, step=10)
        st.sidebar.markdown('''TSNE Learning Rate:''')
        learning_rate = st.sidebar.number_input(hyperParamDescs['TSNE_LR'], value=200, min_value=50, max_value=10000, step=25)
        st.sidebar.markdown('''TSNE Early Exaggeration:''')
        exagg = st.sidebar.number_input(hyperParamDescs['TSNE_EE'], value=12, min_value=0, max_value=10000, step=25)
        if not gpu_avail : 
            st.sidebar.markdown('''TSNE Max Iterations:''')
            max_iterations = st.sidebar.number_input(hyperParamDescs['TSNE_MaxIt'], value=1000, min_value=500, max_value=2000, step=100)
        else :
            max_iterations = 1000
    else :
        st.sidebar.markdown('''UMAP Number of Neighbors:''')
        n_neighbors = st.sidebar.number_input(hyperParamDescs['UMAP_NN'], value=15, min_value=2, max_value=10000, step=10)
        st.sidebar.markdown('''UMAP Minimum Distance:''')
        min_dist = st.sidebar.number_input(hyperParamDescs['UMAP_MinDist'], value=0.1, min_value=0.0, max_value=1.0, step=0.1)
        st.sidebar.markdown('''UMAP Distance Metric:''')
        umap_metrics = ['euclidean','manhattan','chebyshev','minkowski','canberra','braycurtis','mahalanobis','cosine','correlation']
        umap_metric = st.sidebar.selectbox(hyperParamDescs['UMAP_DistMetric'], umap_metrics)
        
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
        st.sidebar.success('File \'{}\' saved!'.format(file_name))

def sessionIDSetup() :
    '''
    This function sets up the session ID logistics and handles manual ID setting.

    Returns
    -------
    datadir : TYPE
        The directory where the data for this session should be stored
    sessionID : TYPE
        The current ID of the streamlit session which can manually specified if 
        correct or randomly assigned.
    debug
        True if you want to output debug info about the directory
    '''
    sessionID = str(getSessionID())
    overrideSessID = st.sidebar.text_input('Session ID override, current is: ' + sessionID, value='')
    
    sessFound = (len(overrideSessID) > 6) and (str(Path(overrideSessID + '_data')) in os.listdir())
    if sessFound :
        sessionID = overrideSessID
    else :
        if len(overrideSessID) > 0 : st.sidebar.text('Session ID not found')
             
    datadir = Path(sessionID + '_data')
    
    # Debug output
    if 'debug' in overrideSessID :
        st.write(os.listdir(), f"\nCurrent data directory: {datadir}")
    
    return datadir, sessionID

##############################
### Main Program Execution ###
##############################

modeOptions = ['Read Me', 'Generate reduced data', 'Plot reduced data']
st.sidebar.image('GECO_logo.jpg', use_column_width=True)
datadir, sessionID = sessionIDSetup()

st.sidebar.header('Select Mode:')
mode = st.sidebar.radio("", modeOptions, index=0)
tabMethods = [readMe, genData, plotData]
tabMethods[modeOptions.index(mode)]()
deleteOldSessionData()
