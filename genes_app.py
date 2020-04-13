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

#%% Check for GPU and try to import TSNECuda
gpu_avail = False
try :
    from tsnecuda import TSNE as TSNECuda        
    # import ctypes    
    # libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    # for libname in libnames:
    #     try:
    #         cuda = ctypes.CDLL(libname)
    #     except OSError:
    #         continue
    #     else:
    #         break
    # else:
    #     gpu_avail = False
    # result = cuda.cuInit(0)
    # assert result == 0
    gpu_avail = True
except : 
    print('TSNE Cuda could not be imported, not using GPU.')

nat_sort = lambda l : ns.natsorted(l)
# nat_sort = lambda l : sorted(l,key=lambda x:int(re.sub("\D","",x) or 0))
removeinvalidchars = lambda s : re.sub('[^a-zA-Z0-9\n\._ ]', '', s).strip()

#%%
# host_dir = st.text_input('Folder path', value='/home/starstorms/Projects/Amber/input data')
# st.write(host_dir)
# st.write(os.path.exists(host_dir))
# try :
#     st.write(os.listdir(host_dir))
# except Exception as e:
#     st.write('Error\n\n', str(e))

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
        ppval = lambda x,y : pearsonr(x,y)[1]
        dfcorr = dfsub.corr(method='pearson')
        dfcorrp = dfsub.corr(method=ppval)
        
        pval_min = 0.01
        labels = pd.DataFrame().reindex_like(dfcorrp)
        labels[dfcorrp >= pval_min] = ''
        labels[dfcorrp < pval_min] = '*'
        np.fill_diagonal(labels.values, '')
        
        g = sns.clustermap(dfcorr, center=0, annot=labels.values, fmt='', linewidths=0.01)
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
    limits = ['Reduced X min', 'Reduced X max', 'Reduced Y min', 'Reduced Y max']
    lims_sel = [st.sidebar.number_input(lim, value=0) for lim in limits]
    lims_all = [-9999, 9999, -9999, 9999]
    lims = lims_all if get_all else lims_sel
        
    selected_genes = dfgene[(dfgene.red_x > lims[0]) & (dfgene.red_x < lims[1]) &
                            (dfgene.red_y > lims[2]) & (dfgene.red_y < lims[3])]
    
    if len(selected_genes) > 0 :
        st.header('List of selected genes in window')
        st.markdown(get_table_download_link(selected_genes), unsafe_allow_html=True)
        st.text(',\n'.join(selected_genes.geneid.values))

def checkDeleteTempVects() :
    try : os.remove('data/temp_dfreduce.csv')
    except : pass

#%% Main Methods
def plotData() :
    header.title('Plotting Gene Data')
    setWideModeHack()
    checkDeleteTempVects()
    checkMakeDataDir()
    
    st.sidebar.header('Load Data')
    csv_files = [os.path.join('data', fp) for fp in os.listdir(os.path.join(os.getcwd(), 'data')) if fp.startswith('dfreduce_') and fp.endswith('.csv')]
    if len(csv_files) == 0 :
        st.write('No files found. Generate files in the Generate reduced data mode.')
        return
    file_names_nice = nat_sort([fp.replace('data/dfreduce_','') for fp in csv_files])
    file_name = 'data/dfreduce_' + st.sidebar.selectbox('Select data file', file_names_nice)
    if st.sidebar.button('Delete this dataset') :
        os.remove(file_name)
        st.success('File \'{}\' removed, please select another file.'.format(file_name.replace('data/dfreduce_', '')))
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
    header.title('Streamlit App Manual')
    st.markdown("""
            This streamlit app allows for generating and viewing reduced dimensionality plots for various data.
            
            ## File Requirements ##
            - First column must contain the list of genes, it will be renamed 'geneid'.
            - No negative numbers or previous transformations (like log or normalization), use raw information.
            - Blank gene expression example file here
            - Blank gene marker example file here
            - No index column in csv (1, 2, 3, ...)
            - 'NA' entries will be interpreted as unknowns and removed
                - Also any of: ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', 'NA', '', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']
            
            ## Usage Instructions ##
            1. Start on the 'Generate reduced data' mode
                - See here for info on setting TSNE parameters: <a href="https://distill.pub/2016/misread-tsne/">Guide on setting TSNE parameters</a>
            2. Save the data
            3. Switch to 'Plot data' mode
            
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

def umapReduce(npall, n_neighbors=15, min_dist=0.1, metric='euclidean') :
    print('Running UMAP')
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
    return tsne.fit_transform(princcomps)

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
    ralgo = st.sidebar.selectbox('Reduction algorightm:', ['TSNE', 'UMAP'])
    useUmap = ralgo == 'UMAP'    

    param_guide_links = ['https://distill.pub/2016/misread-tsne/', 'https://umap-learn.readthedocs.io/en/latest/parameters.html']
    st.sidebar.markdown('<a href="{}">Guide on setting {} parameters</a>'.format(param_guide_links[useUmap], ralgo), unsafe_allow_html=True)
    remove_zeros = st.sidebar.checkbox('Remove entries with all zeros?', value=True)    
    norm_per_row = st.sidebar.checkbox('Normalize per gene (row)?', value=True) if len(avg_cols) > 2 else False
    norm_control = st.sidebar.checkbox('Normalize to control?', value=False)
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
        dfreduce.round(decimals=4).to_csv('data/temp_dfreduce.csv', index=False)
    elif not os.path.exists('data/temp_dfreduce.csv') :
        return
    else :
        status = st.header('Loading previous vectors')
        dfreduce = pd.read_csv('data/temp_dfreduce.csv')
        
    st.sidebar.header('Plot Quick View Options')
    chosen_color = st.sidebar.selectbox('Color data', ['Type'] + all_types)
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
        ax.set_xlabel('red_x')
        ax.set_ylabel('red_y')
    
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
        
    file_name = removeinvalidchars(st.sidebar.text_input('Data file name:', value=suggested_fn))   
    if len(file_name) > 0 and st.sidebar.button('Save data file') :
        
        dfsave = dfgene.copy()
        dfsave = pd.merge(dfsave, dfreduce[['geneid', 'red_x', 'red_y']], on='geneid', how='right')
        dfsave = dfsave.round(decimals=3)
        checkMakeDataDir()
        dfsave.to_csv('data/dfreduce_' + file_name + '.csv', index=False)
        st.success('File \'{}\' saved!'.format(file_name))

#%% Main program execution
modeOptions = ['Read Me', 'Generate reduced data', 'Plot reduced data']
st.sidebar.header('Select Mode:')
mode = st.sidebar.radio("", modeOptions, index=0)
tabMethods = [readMe, genData, plotData]
tabMethods[modeOptions.index(mode)]()