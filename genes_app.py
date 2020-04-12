#%% Imports
import streamlit as st
header = st.title("Starting up...")
import numpy as np
import os
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re
from collections import Counter
import plotly.express as px
import base64
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
import natsort as ns

#%% Check for GPU and try to import TSNECuda
gpu_avail = False
try :
    from tsnecuda import TSNE as TSNECuda
    gpu_avail = True
except : pass

nat_sort = lambda l : ns.natsorted(l)
# nat_sort = lambda l : sorted(l,key=lambda x:int(re.sub("\D","",x) or 0))
removeinvalidchars = lambda s : re.sub('[^a-zA-Z0-9\n\._ ]', '', s).strip()

#%% Helper Methods
def setWideModeHack():
    max_width_str = f"max-width: 1200px;"
    st.markdown( f""" <style> .reportview-container .main .block-container{{ {max_width_str} }} </style> """, unsafe_allow_html=True)

def get_table_download_link(df):
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
    if len1 != len4 :
        warning_messages.append('\nTotal entries remaining after cleaning: {}'.format(len4))
    
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

    # this list comprehension should work but Streamlit has a bug and can't hash this type of thing correctly so it has to be done in a ridiculously overcomplicated way...
    # typecols = [[col for col in list(dfgene.columns) if col.startswith(typ)] for typ in np.array(all_types)]        
    typecols = [list() for i in range(len(all_types))]
    for col in dfgene.columns :
        for prefix in all_types :
            if col.rsplit('_', 1)[0] == prefix :
                typecols[all_types.index(prefix)].append(col)
    
    for i in range(len(typecols)) :            
        dfgene['avg_'+all_types[i]] = (dfgene.loc[:,typecols[i]].mean(axis=1)).round(decimals=9).astype(float)
        
    avg_cols = [col for col in dfgene.columns if col.startswith('avg_')]    
    dfgene['type'] = dfgene[avg_cols].idxmax(axis=1).str.slice(start=4)
    dfgene['avg_type'] = dfgene.apply(lambda row : row['avg_' + row.type], axis=1)
    
    avg_cols = nat_sort(avg_cols)
    checkDeleteTempVects()
    return dfgene, all_types, sample_names, typecols, avg_cols, warning_messages

# This methods ensures evey column header ends in a '_X' where X is the sample number
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
    color_indices = all_types + ['Type', 'Type Average', 'Normalized to control (select)']
    color_nums = ['avg_'+typ for typ in all_types] + ['type', 'avg_type', 'norm_control']
    chosen_color = color_nums[ color_indices.index(st.sidebar.selectbox('Color Data', color_indices))]
    if chosen_color == 'norm_control' :
        control = st.sidebar.selectbox('Select control:', all_types)
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
    xmin, xmax = dfgene.tsne1.min(), dfgene.tsne1.max()
    ymin, ymax = dfgene.tsne2.min(), dfgene.tsne2.max()
    xlims, ylims = [xmin*padding, xmax*padding], [ymin*padding, ymax*padding]
    return xlims, ylims

def plotTSNE(dfgene, all_types, color_scale, chosen_color, gids_found, markers_found, xlims, ylims, log_color_scale) :
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
    
    fig = px.scatter(dfgene, x="tsne1", y="tsne2",
                     color=color_log,
                     range_x = xlims, range_y = ylims,
                     width=1200, height=800,
                     hover_name='geneid',
                     hover_data= nat_sort([col for col in dfgene.columns if col.startswith('avg_')]),
                     category_orders={"type": all_types},
                     color_continuous_scale = color_scale,
                     color_discrete_sequence = color_disc_seq )
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
        fig.add_scatter(name='Genes', text=gids_found.geneid.values, mode='markers', x=gids_found.tsne1.values, y=gids_found.tsne2.values, line=dict(width=5), marker=dict(size=20, opacity=1.0, line=dict(width=3)) )
    
    if len(markers_found) > 0 :
        fig.add_scatter(name='Markers', text=markers_found.geneid.values, mode='markers', x=markers_found.tsne1.values, y=markers_found.tsne2.values, line=dict(width=1), marker=dict(size=20, opacity=1.0, line=dict(width=3)) )
    
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
        ppval = lambda x,y : pearsonr(x,y)[1]
        dfcorr = dfsub.corr(method='pearson')
        dfcorrp = dfsub.corr(method=ppval)
        
        pval_min = 0.01
        labels = pd.DataFrame().reindex_like(dfcorrp)
        labels[dfcorrp >= pval_min] = ''
        labels[dfcorrp < pval_min] = '*'
        np.fill_diagonal(labels.values, '')
        
        g = sns.clustermap(dfcorr, center=0, annot=labels.values, fmt='', square=True, linewidths=0.01)
        st.pyplot()

    if len(gids_found) > 1 :
        fig, axes = plt.subplots(1, 1)
        dfsub_norm = dfsub / dfsub.sum()
        normalize = st.sidebar.checkbox('Normalize heatmap?', value=True)    
        dfheatmap = dfsub.T if not normalize else dfsub_norm.T
        
        g = sns.clustermap(dfheatmap, metric="correlation", method="single", cmap="Blues",
                            standard_scale=1, col_cluster=False,
                            figsize=(10, 3 + 0.25 * len(gids_found)), linewidths=0.2)
        g.cax.set_visible(False)
        axes.set_ylabel('')    
        axes.set_xlabel('')
        plt.subplots_adjust(top=1.2, right=0.85, bottom=.2, hspace=0.0)    
        st.pyplot()
    
def selectGenes(dfgene) :
    get_all = st.sidebar.button('Get all genes')
    limits = ['TSNE X min', 'TSNE X max', 'TSNE Y min', 'TSNE Y max']
    lims_sel = [st.sidebar.number_input(lim, value=0) for lim in limits]
    lims_all = [-9999, 9999, -9999, 9999]
    lims = lims_all if get_all else lims_sel
        
    selected_genes = dfgene[(dfgene.tsne1 > lims[0]) & (dfgene.tsne1 < lims[1]) &
                            (dfgene.tsne2 > lims[2]) & (dfgene.tsne2 < lims[3])]
    
    if len(selected_genes) > 0 :
        st.header('List of selected genes in window')
        st.markdown(get_table_download_link(selected_genes), unsafe_allow_html=True)
        st.text(',\n'.join(selected_genes.geneid.values))

def checkDeleteTempVects() :
    try : os.remove('data/temp_dftsne.csv')
    except : pass

#%% Main Methods
def plotData() :
    header.title('Plotting Gene Data')
    setWideModeHack()
    checkDeleteTempVects()
    checkMakeDataDir()
    
    st.sidebar.header('Load Data')
    csv_files = [os.path.join('data', fp) for fp in os.listdir(os.path.join(os.getcwd(), 'data')) if fp.startswith('dftsne_') and fp.endswith('.csv')]
    if len(csv_files) == 0 :
        st.write('No files found. Generate files in the Generate TSNE data mode.')
        return
    file_names_nice = nat_sort([fp.replace('data/dftsne_','') for fp in csv_files])
    file_name = 'data/dftsne_' + st.sidebar.selectbox('Select TSNE data file', file_names_nice)
    if st.sidebar.button('Delete this dataset') :
        os.remove(file_name)
        st.success('File \'{}\' removed, please select another file.'.format(file_name.replace('data/dftsne_', '')))
        return
    
    # Load Data
    dfgene, all_types, sample_names, avg_cols = getDataPlot(file_name)
    dfplot = dfgene.copy(deep=True)
    
    # Get Inputs
    st.sidebar.header('TSNE Plot View Parameters')
    chosen_color, norm_control = askColor(all_types)    
    log_color_scale = False if chosen_color == 'type' else st.sidebar.checkbox('Log color scale?', value=True)
    color_scale = askColorScale(chosen_color)
    
    if chosen_color == 'norm_control' and norm_control :
        min_val = dfplot[avg_cols].min().min()
        dfplot['norm_control'] = (dfplot[avg_cols].div(dfplot[avg_cols].sum(axis=1), axis=0)[norm_control] * 100).round(1)
    
    avg_typ_min, avg_typ_max = min(0.0,float(dfplot.avg_type.min())), float(dfplot.avg_type.max())
    min_expression_typ = st.sidebar.number_input('Min assigned type expression to show:', min_value = avg_typ_min, value = avg_typ_min, max_value = avg_typ_max, step=0.1)
    
    if not (chosen_color == 'type' or chosen_color == 'avg_type') :
        avg_sel_min, avg_sel_max = min(0.0,float(dfplot[chosen_color].min())),  float(dfplot[chosen_color].max())
        # sel_min_title = 'Min normalized % of {} to show'.format(norm_control[4:]) if chosen_color == 'norm_control' else 'Min expression of {} to show'.format(chosen_color[4:])
        sel_min_title = 'Min normalized % of control to show' if chosen_color == 'norm_control' else 'Min expression of {} to show:'.format(chosen_color[4:])
        min_expression_sel = None if (chosen_color == 'type' or chosen_color == 'avg_type') else st.sidebar.number_input(sel_min_title, min_value = avg_sel_min, value = avg_sel_min, max_value=avg_sel_max, step=5.0 if chosen_color == 'norm_control' else 0.1)

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
    
    # Main TSNE plot
    plotTSNE(dfplot, all_types, color_scale, chosen_color, gids_found, markers_found, xlims, ylims, log_color_scale)
    
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
    header.title('Streamlit App Manual')
    st.markdown("""
            This streamlit app allows for generating and viewing TSNE plots for various data.
            
            ## File Requirements ##
            - First column must contain the list of genes, it will be renamed 'geneid'.
            - No negative numbers or previous transformations (like log or normalization), use raw information.
            - Blank gene expression example file here
            - Blank gene marker example file here
            - No index column in csv (1, 2, 3, ...)
            - 'NA' entries will be interpreted as unknowns and removed
                - Also any of: ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', 'NA', '', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']
            
            ## Usage Instructions ##
            1. Start on the 'Generate TSNE data' mode
                - See here for info on setting TSNE parameters: <a href="https://distill.pub/2016/misread-tsne/">Guide on setting TSNE parameters</a>
            2. Save the data
            3. Switch to 'Plot TSNE data' mode
            
            ### Notes: ###
            - Gene IDs must match exactly (case sensitive)
            - If you run into odd persistent errors hit 'c' and clear cache and then hit 'r' to reload.
            
            ### Suggested color scales: ###
            - Blackbody
            - Electric
            - Jet
            - Thermal
            
            """, unsafe_allow_html=True )

def checkMakeDataDir() :
    if os.path.exists('data') :
        return
    os.mkdir('data')

def reduce(npall, pca_components=0, perp=40, learning_rate=200, n_iter=1000) :    
    if pca_components > 0 :
        pca = PCA(n_components=pca_components)
        princcomps = pca.fit_transform(npall)
    else :
        princcomps = npall
    
    if gpu_avail :
        try :
            tsne_vects_out = TSNECuda(n_components=2, perplexity=perp, learning_rate=learning_rate).fit_transform(princcomps)
            return tsne_vects_out
        except :
            pass
    tsne = TSNE(n_components=2, n_iter=n_iter, verbose=3, perplexity=perp, learning_rate=learning_rate)
    return tsne.fit_transform(princcomps)

def genTSNE() :
    header.title('Generate TSNE Data')
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
    
    st.sidebar.header('TSNE Run Parameters')
    st.sidebar.markdown('<a href="https://distill.pub/2016/misread-tsne/">Guide on setting TSNE parameters</a>', unsafe_allow_html=True)
    remove_zeros = st.sidebar.checkbox('Remove entries with all zeros?', value=True)    
    norm_per_row = st.sidebar.checkbox('Normalize per gene (row)?', value=True) if len(avg_cols) > 2 else False
    norm_control = st.sidebar.checkbox('Normalize to control?', value=False)
    if norm_control :
        control = st.sidebar.selectbox('Select control:', all_types)
    
    pca_comp = st.sidebar.number_input('PCA components (0 to run only TSNE)', value=0, min_value=0, max_value=len(all_types)-1, step=1)
    perp = st.sidebar.number_input('TSNE Perplexity', value=50, min_value= 40 if gpu_avail else 2, max_value=10000, step=10)
    learning_rate = st.sidebar.number_input('TSNE Learning Rate', value=200, min_value=50, max_value=10000, step=25)
    if not gpu_avail : max_iterations = st.sidebar.number_input('TSNE Max Iterations', value=1000, min_value=500, max_value=2000, step=100)
    else : max_iterations = -1
    
    if st.sidebar.button('Run TSNE reduction') :
        status = st.header('Running TSNE reduction')        
        dftsne = dfgene.copy(deep=True)
        
        if remove_zeros :
            dftsne = dftsne.loc[(dftsne[avg_cols]!=0).any(axis=1)]        
        if norm_control or norm_per_row :
            dftsne[avg_cols] = dftsne[avg_cols] + sys.float_info.epsilon
        if norm_control :
            dftsne[avg_cols] = dftsne[avg_cols].div(dftsne['avg_'+control], axis=0)
        if norm_per_row :
            dftsne[avg_cols] = dftsne[avg_cols].div(dftsne[avg_cols].sum(axis=1), axis=0)
        if norm_control or norm_per_row :
            dftsne[avg_cols] = dftsne[avg_cols].round(decimals=4)
        
        if (dftsne[avg_cols].isna().sum().sum() > 0) :
            st.write('!Warning! Some NA values found in data, removed all entries with NAs, see below:', dftsne[avg_cols].isna().sum())
            dftsne = dftsne.dropna()
        
        tsne_vects_in = dftsne[avg_cols].values + sys.float_info.epsilon
        lvects = reduce(tsne_vects_in, pca_components=pca_comp, perp=perp, learning_rate=learning_rate, n_iter=max_iterations)
        dftsne['tsne1'] = lvects[:,0]
        dftsne['tsne2'] = lvects[:,1]
        checkMakeDataDir()
        dftsne.round(decimals=4).to_csv('data/temp_dftsne.csv', index=False)
    elif not os.path.exists('data/temp_dftsne.csv') :
        return
    else :
        status = st.header('Loading previous vectors')
        dftsne = pd.read_csv('data/temp_dftsne.csv')
        
    st.sidebar.header('TSNE Quick View Options')
    chosen_color = st.sidebar.selectbox('Color data', ['Type'] + all_types)
    hue = 'type' if chosen_color == 'Type' else 'avg_' + chosen_color
    
    if chosen_color == 'Type' :
        ax = sns.scatterplot(data=dftsne, x='tsne1', y='tsne2', s=5, linewidth=0.01, hue=hue)
        ax.set(xticklabels=[], yticklabels=[], xlabel='tsne1', ylabel='tsne2')
        plt.subplots_adjust(top=0.98, left=0.05, right=1, bottom=0.1, hspace=0.0)
    else :
        fig, ax = plt.subplots(1)
        plt.scatter(x=dftsne.tsne1.values, y=dftsne.tsne2.values, s=5, linewidth=0.01, c=dftsne[hue].values, norm=matplotlib.colors.LogNorm())
        plt.colorbar(label='Expression Level')
        plt.subplots_adjust(top=0.98, left=0.05, right=1, bottom=0.1, hspace=0.0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlabel('tsne1')
        ax.set_ylabel('tsne2')
    
    status.header('TSNE data quick view:')
    st.pyplot()
    st.write('Total number of points: ', len(dftsne))
    
    st.sidebar.header('TSNE Data Save')
    suggested_fn = '{}p'.format(perp)
    suggested_fn += '_No0' if remove_zeros else ''
    suggested_fn += '_NR' if norm_per_row else ''
    suggested_fn += '_NC-'+control if norm_control else ''
    file_name = removeinvalidchars(st.sidebar.text_input('Data file name:', value=suggested_fn))   
    print(file_name)
    if len(file_name) > 0 and st.sidebar.button('Save TSNE data') :
        
        dfsave = dfgene.copy()
        dfsave = pd.merge(dfsave, dftsne[['geneid', 'tsne1', 'tsne2']], on='geneid', how='right')
        dfsave = dfsave.round(decimals=3)
        checkMakeDataDir()
        dfsave.to_csv('data/dftsne_' + file_name + '.csv', index=False)
        st.success('File \'{}\' saved!'.format(file_name))

#%% Main program execution
modeOptions = ['Read Me', 'Generate TSNE data', 'Plot TSNE data']
st.sidebar.header('Select Mode:')
mode = st.sidebar.radio("", modeOptions, index=0)
tabMethods = [readMe, genTSNE, plotData]
tabMethods[modeOptions.index(mode)]()