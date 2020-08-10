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
