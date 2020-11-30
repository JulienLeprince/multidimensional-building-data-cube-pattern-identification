# Import modules
import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
# SAX package - source https://github.com/seninp/saxpy
from saxpy.alphabet import cuts_for_asize
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.paa import paa
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Plotting modules
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
plt.rcdefaults()
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.colors import n_colors
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True)
import plotly.io as pio


########################################################################################
###                            Pre-Processing functions                              ###
########################################################################################

def reduce_mem_usage(df, verbose=True):
    """"Function to reduce the memory usage of a dataframe.
    Source: https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction"""

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


########################################################################################
###                                Pre-Mining functions                              ###
########################################################################################

### Data Selection functions

def multicol_2ndColumnSelection(df_multicol, allcol1, col2):
    """"Function to select data from a multi-column dataframe based on the 2nd column value.
    From a defined 2nd-level column of interest - col2,
     the function loops over the dataframe from all the values interest from the 1st-level column - allcol1"""
    df = pd.DataFrame()
    for i in allcol1:
        df[i] = df_multicol[i, col2].copy()
    return df

def multi2singlecol_1stCol(df_in):
    """"Function to transform a 2 column dataframe to a single one, while appending the 2nd column information
    to a new attribute."""
    # Extract upper level column meter_type information
    meter_type_list = []
    for meter_type, blg_id in df_in.columns.values:
        meter_type_list.append(meter_type)
    meter_type_list = list(set(meter_type_list))

    dfs = []
    for i in meter_type_list:
        df1 = pd.melt(df_in[i].reset_index(),
                      id_vars=df_in.index.name,
                      var_name="building_id",
                      value_name=i)
        df1.set_index(["building_id", df_in.index.name], inplace=True)
        dfs.append(df1)  # append to list
    meter_df = pd.concat(dfs, axis=1)
    meter_df = meter_df.reset_index().set_index([df_in.index.name], drop=True)
    return meter_df

def multicol_inverseCols(df_multicol):
    """"Function to inverse the order of a multicolumn dataframe."""
    tuple_inv_column = []
    for col1, col2 in df_multicol.columns.values:
        tuple_inv_column.append((col2, col1))

    # Create new dataframe from inversed multicolumn
    multi_index = pd.MultiIndex.from_tuples(tuple_inv_column)
    df_inv = pd.DataFrame(columns=multi_index)
    for col1, col2 in df_multicol.columns.values:
        df_inv[(col2, col1)] = df_multicol[(col1, col2)]
    return df_inv

def checkIfExists(elem):
    if elem:
        return elem
    else:
        return ['None']

def flatten_list(l):
    for el in l:
        if isinstance(el, list):
            yield from flatten_list(el)
        else:
            yield el

### Scaling functions

def scale_NanRobust(data_array, scaler):
    """ A function to scale an array while being robust to outliers.
    Adapted from: https://stackoverflow.com/questions/55280054/handling-missing-nan-values-on-sklearn-preprocessing"""
    # Set valid mask
    nan_mask = np.isnan(data_array)
    valid_mask = ~nan_mask
    # create a result array
    result = np.full(data_array.shape, np.nan)
    # assign only valid cases to
    result[valid_mask] = scaler.fit_transform(data_array[valid_mask].reshape(-1, 1)).reshape(data_array[valid_mask].shape)
    return result

def scale_df_columns_NanRobust(df_in, target_columns, scaler=MinMaxScaler(feature_range=(1, 2))):
    """"A function to normalize columns of a dataframe per column, while being robust to Nan values.
    The function returns a similar dataframe with missing values in identical places - normalized with the scaler object."""
    # Identify target from non-target column values
    nontarget_columns = list(set(df_in.columns) - set(target_columns))
    df = df_in[target_columns].copy()
    # Scale over the target columns
    array_scaled = []
    for col in df.columns:
        array_scaled.append(scale_NanRobust(df[col].values, scaler))
    df_scaled = pd.DataFrame(np.vstack(array_scaled).transpose(), columns=df.columns)
    # Set scaled dataframe index
    df_scaled[df_in.index.name] = df_in.index
    df_scaled.set_index([df_in.index.name], inplace=True, drop=True)
    # Merge non-target columns to the scaled frame
    df_scaled[nontarget_columns] = df_in[nontarget_columns]
    return df_scaled


### SAX transformation functions

def SAX_mining(df_in, W=4, A=3):
    """"Function to perform daily SAX mining on input dataframe"""
    # Input definition of the function
    df_sax = df_in.copy()
    df_sax['Day'] = df_sax.index.dayofyear
    df_sax['Hour'] = df_sax.index.hour

    # Daily SAX over the year with reduced daily size
    sax_dict, counts, sax_data = dict(), dict(), dict()
    for meter in df_in.columns.values:
        # Daily heatmaps over all year
        sax_data[meter] = pd.pivot_table(df_sax, values=meter,
                                  index=['Day'], columns='Hour')
        sax_data[meter] = reduce_mem_usage(sax_data[meter], verbose=False)

        # Daily SAX obtained here with hourly resolution
        daily_sax = []
        for i in range(sax_data[meter].shape[0]):
            dat_paa = paa(sax_data[meter].values[i], W)
            daily_sax.append(ts_to_string(dat_paa, cuts_for_asize(A)))
        sax_dict[meter] = daily_sax

        # Now count the number of similar elements in the SAX list
        counts[meter] = Counter(sax_dict[meter])
    return sax_dict, counts, sax_data

### Data formating functions

def sax_count_reformat(sax_dict):
    """"Function to format SAX counts to a unified dataframe."""
    list_concat = []
    counts = dict()
    for meter_data in sax_dict:
        counts[meter_data] = Counter(sax_dict[meter_data])
        # Create a dataframe from the counter object
        list_concat.append(pd.DataFrame.from_dict(counts[meter_data], orient='index', columns=[meter_data]))
    # Now concatenate the dictionary to one dataframe
    df_count = pd.concat(list_concat, axis=1)  # Reformated dataframe
    return df_count

def sax_df_reformat(sax_data, sax_dict, meter_data, space_btw_saxseq=3):
    """"Function to format a SAX timeseries original data for SAX heatmap plotting."""

    counts_nb = Counter(sax_dict[meter_data])
    # Sort the counter dictionnary per value
    # source: https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    counter = {k: v for k, v in sorted(counts_nb.items(), key=lambda item: item[1])}
    keys = counter.keys()

    empty_sax_df = pd.DataFrame(columns=sax_data[meter_data].columns, index=[' ']*space_btw_saxseq)
    new_sax_df = pd.DataFrame(columns=sax_data[meter_data].columns)
    for sax_seq in keys:
        # Obtaining sax indexes of corresponding profiles within dataframe
        indexes = [i for i,x in enumerate(sax_dict[meter_data]) if x == sax_seq]   # returns all indexes
        # Formating a newdataframe from selected sax_seq
        df_block = sax_data[meter_data].iloc[indexes].copy()
        df_block["SAX"] = [sax_seq]*len(indexes)
        new_sax_df = pd.concat([df_block, empty_sax_df, new_sax_df], axis=0) # Reformated dataframe
    # Mapping the sax sequence to the data
    index_map_dictionary = dict()
    index_map_dictionary["SAX_seq"], index_map_dictionary["SAX_idx"] = [], []
    for sax_seq in keys:
        indexes = [i for i, x in enumerate(new_sax_df["SAX"]) if x == sax_seq]  # returns all indexes
        index_map_dictionary["SAX_seq"].append(sax_seq)
        index_map_dictionary["SAX_idx"].append(np.median(indexes))
    # Droping the SAX column of the dataframe now that we have a mapping variable for it
    new_sax_df.drop("SAX", axis=1, inplace=True)
    return new_sax_df, index_map_dictionary

def clust_df_reformat(df_data, clust_dict, meter_data, space_btw_saxseq=3, s_size=5):
    """"Function to format the timeseries original data for clustered heatmap plotting."""

    # Initializing new dataframes
    empty_sax_df = pd.DataFrame(columns=df_data.index, index=[' ']*space_btw_saxseq)
    new_sax_df = pd.DataFrame(columns=df_data.index)
    # Creating quantiles vector
    qtl1 = list(np.linspace(0.25, 0.5, 2+s_size))
    qtl2 = list(np.linspace(0.5, 0.75, 2+s_size))
    qtls = list(dict.fromkeys(qtl1 + qtl2))

    for clus in clust_dict:
        try:
            # Selecting cluster data per attribute from frame
            df_c = multicol_inverseCols(df_data[clust_dict[clus]])[meter_data]
            # Normalizing
            df_c = scale_df_columns_NanRobust(df_c, df_c.columns, scaler=StandardScaler())
            # Calculating condensed quantile information for display
            df_block = df_c.transpose().quantile(qtls)
        except KeyError:
            # If column does not exist in DataFrame, i.e. droped from all Nans, use an empty frame instead
            df_block = pd.DataFrame(columns=df_data.index, index=[' ']*len(qtls))
        # Formating a newdataframe from selected sax_seq
        df_block["cluster_id"] = [clus]*len(df_block.values)
        new_sax_df = pd.concat([df_block, empty_sax_df, new_sax_df], axis=0) # Reformated dataframe

    # Mapping the sax sequence to the data
    index_map_dictionary = {"clust_idx": [], "clust": []}
    for clus in clust_dict:
        indexes = [i for i, x in enumerate(new_sax_df["cluster_id"]) if x == clus]  # returns all indexes
        index_map_dictionary["clust"].append("cluster "+str(clus)+ " (N=" + str(len(clust_dict[clus]))+ ")")
        index_map_dictionary["clust_idx"].append(np.median(indexes))
    # Droping the SAX column of the dataframe now that we have a mapping variable for it
    new_sax_df.drop("cluster_id", axis=1, inplace=True)
    # Keeping only Hourly information from datetime columns
    new_sax_df.columns = pd.to_datetime(new_sax_df.transpose().index).hour
    return new_sax_df, index_map_dictionary

### Visualization functions functions

def SAXcount_hm_wdendro(df_count, title):
    # Create Side Dendrogram
    # source: https://plotly.com/python/dendrogram/
    dendo = ff.create_dendrogram(df_count.values, orientation='left', labels=list(df_count.index.values))
    for i in range(len(dendo['data'])):
        dendo['data'][i]['xaxis'] = 'x2'
    # Create Heatmap
    dendro_leaves_txt = dendo['layout']['yaxis']['ticktext']
    # Convert the txt leaves to integer index values
    dendro_leaves = []
    for txt in df_count.index.values:
        dendro_leaves.append(list(dendro_leaves_txt).index(txt))
    dendro_leaves = list(map(int, dendro_leaves))

    heat_data = df_count.values[dendro_leaves, :]

    # Calling the subplots
    fig = go.Figure(data=go.Heatmap(z=heat_data,
                                    x=df_count.columns,
                                    y=df_count.index[dendro_leaves],
                                    # zmax=pzmax, zmin=pzmin,
                                    colorbar={"title": "Counts"},
                                    colorscale='Blues'))
    p_width = len(df_count.columns)*5 if len(df_count.columns)*5 > 400 else 400
    fig.update_layout(height=900, width=p_width,
                      xaxis={"tickmode": "array"},
                      title_text=f"SAX counts for attribute: {title}",
                      plot_bgcolor='#fff'
                      )
    return fig

def counter_plot(counter, title=None):
    """Simple demo of a horizontal bar chart.
    Source: https://stackoverflow.com/questions/22222573/how-to-plot-counter-object-in-horizontal-bar-chart"""
    # Sort the counter dictionnary per value
    # source: https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}
    # Counter data, counter is your counter object
    keys = counter.keys()
    y_pos = np.arange(len(keys))
    # get the counts for each key, assuming the values are numerical
    performance = [counter[k] for k in keys]
    # Now plotting
    fig = plt.figure(figsize=(6, 6))
    plt.barh(y_pos, performance, align='center', alpha=0.4)
    plt.yticks(y_pos, keys)
    plt.xlabel('Number of profiles per symbolic sequence')
    plt.title(title)
    plt.show()
    return fig

def SAX_dailyhm_visualization(dict_numeric, sax_dict, index_map_dictionary, title):
    """"Function to visualize multi-attribute SAX sequences from original dataframe with SAX dictionary"""
    keys = list(sax_dict.keys())
    key_int = 0
    # First loop over all keys and data to identify min and max values of the time series
    for key in sax_dict:    # key can be meter_data or bld_id depending on the cuboid selected
        sax_seq_int = 0
        for sax_sequence_toidentify in sax_dict[key]:
            indexes = [i for i, x in enumerate(sax_dict[key]) if x == sax_sequence_toidentify]  # returns all indexes

            if key_int < 1:
                pzmax = dict_numeric[key].iloc[indexes].max().max()
                pzmin = dict_numeric[key].iloc[indexes].min().min()
            else:
                pzmax = max(pzmax, dict_numeric[key].iloc[indexes].max().max())
                pzmin = min(pzmin, dict_numeric[key].iloc[indexes].min().min())
            sax_seq_int = sax_seq_int + 1
        key_int = key_int + 1

    # Calling the subplots
    fig = make_subplots(rows=1, cols=len(keys), shared_yaxes=False, 
    horizontal_spacing=0.01+len(sax_dict[key][0])*0.005, column_titles=keys, x_title="Hour of the day")
    # Then Loop again of the set to plot
    key_int = 0
    # Looping over sax keys (i.e. attributes or blg keys)
    for key in sax_dict:
        # Plot
        fig.add_trace(go.Heatmap(z=dict_numeric[key],
                                 x=dict_numeric[key].columns,
                                 zmax=pzmax, zmin=pzmin,
                                 colorbar={"title": "Attribute normalized value"},
                                 colorscale='temps'),
                      row=1, col=key_int + 1)
        fig.update_yaxes(tickmode='array',
                         tickvals=index_map_dictionary[key]["SAX_idx"],
                         ticktext=index_map_dictionary[key]["SAX_seq"],
                         row=1, col=key_int+1)
        key_int = key_int + 1
    fig.update_layout(height=800, width=len(keys)*250,
                      xaxis={"tickmode": "array"},
                      title_text=f"Daily SAX profiles of {title}",
                      plot_bgcolor='#fff'
                      )
    return fig

def clust_dailyhm_visualization(dict_numeric, index_map_dictionary, title):
    """"Function to visualize multi-attribute clustered timeseries from original dataframe with cluster dictionary"""
    keys = list(dict_numeric.keys())
    key_int = 0
    # First loop over all keys and data to identify min and max values of the time series
    for key in dict_numeric:    # key can be meter_data or bld_id depending on the cuboid selected
        if key_int < 1:
            pzmax = dict_numeric[key].max().max()
            pzmin = dict_numeric[key].min().min()
        else:
            pzmax = max(pzmax, dict_numeric[key].max().max())
            pzmin = min(pzmin, dict_numeric[key].min().min())
        key_int = key_int + 1
    
    # Calling the subplots
    fig = make_subplots(rows=1, cols=len(keys), shared_yaxes=False, 
    horizontal_spacing=0.005, column_titles=keys, x_title="Hour of the day")
    # Then Loop again of the set to plot
    key_int = 0
    # Looping over sax keys (i.e. attributes or blg keys)
    for key in dict_numeric:
        # Plot
        fig.add_trace(go.Heatmap(z=dict_numeric[key],
                                 x=dict_numeric[key].columns,
                                 zmax=pzmax, zmin=pzmin,
                                 colorbar={"title": "Attribute normalized value"},
                                 colorscale='temps'),
                      row=1, col=key_int + 1)
        fig.update_yaxes(tickmode='array',
                         tickvals=index_map_dictionary[key]["clust_idx"],
                         ticktext=[' ']*len(index_map_dictionary[key]["clust_idx"]),
                         row=1, col=key_int+1)
        key_int = key_int + 1
    fig.update_yaxes(tickmode='array',
                    tickvals=index_map_dictionary[key]["clust_idx"],
                    ticktext=index_map_dictionary[key]["clust"],
                    row=1, col=1)
    fig.update_layout(height=800, width=len(keys)*250,
                      xaxis={"tickmode": "array"},
                      title_text=f"Daily SAX profiles of {title}",
                      plot_bgcolor='#fff')
    return fig

def genSankey(df,cat_cols=[],value_cols='',title='Sankey Diagram'):
    """Wrapper function for Sankey Diagram.
    Source: https://medium.com/kenlok/how-to-create-sankey-diagrams-from-dataframes-in-python-e221c1b4d6b0"""
  
    # maximum of 6 value cols -> 6 colors
    colorPalette = sns.color_palette("Spectral", len(cat_cols)).as_hex()
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp =  list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp
        
    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))
    
    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]]*colorNum
        
    # transform df into a source-target pair
    for i in range(len(cat_cols)-1):
        if i==0:
            sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            sourceTargetDf.columns = ['source','target','count']
        else:
            tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            tempDf.columns = ['source','target','count']
            sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'count':'sum'}).reset_index()
        
    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
    
    # creating the sankey diagram
    data = dict(
        type='sankey',
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = labelList,
          color = colorList
        ),
        link = dict(
          source = sourceTargetDf['sourceID'],
          target = sourceTargetDf['targetID'],
          value = sourceTargetDf['count']
        )
      )
    
    layout =  dict(
        title = title,
        font = dict(
          size = 10
        )
    )
       
    fig = dict(data=[data], layout=layout)
    return fig

def SAXannotated_heatmap_viz(df_olap_norm, df_text, title):
    fig = ff.create_annotated_heatmap(z=df_olap_norm.values.tolist(),
                                      x=df_olap_norm.columns.values.tolist(),
                                      y=df_olap_norm.index.values.tolist(),
                                      annotation_text=df_text,
                                      colorbar={"title": "Counts"},
                                      colorscale='Blues')
    p_height = len(df_olap_norm.index)*15 if len(df_olap_norm.index)*15 > 400 else 400
    p_width = len(df_olap_norm.columns)*120 if len(df_olap_norm.columns)*120 > 400 else 400
    fig.update_layout(height=p_height, width=p_width,
                      xaxis={"tickmode": "array"},
                      plot_bgcolor='#fff'
                      )
    return fig

def SAXannotated_hm_wcounts(df_olap_normalized, df_text, title):
    """Heatmap annoted plot with counts per groupby object"""

    # Drop row duplicates
    df_olap_norm_reduced = df_olap_normalized.loc[df_text.index]

    # Heatmap
    fig1 = ff.create_annotated_heatmap(z=df_olap_norm_reduced.values.tolist(),
                                    x=df_olap_norm_reduced.columns.values.tolist(),
                                    y=df_olap_norm_reduced.index.values.tolist(),
                                    annotation_text=df_text.drop(['count', 'all_buildings'], axis=1).values.tolist(),
                                    colorbar={"title": "Counts"},
                                    colorscale='Blues')
    p_height = len(df_olap_normalized.index)*15 if len(df_olap_normalized.index)*15 > 400 else 400
    p_width = len(df_olap_normalized.columns)*120 if len(df_olap_normalized.columns)*120 > 400 else 400
    fig1.update_layout(height=p_height, width=p_width,
                        xaxis={"tickmode": "array"},
                        #title_text=f" Daily SAX for time slice: {title}",
                        plot_bgcolor='#fff')
    # Barplot
    fig2 = go.Figure(go.Bar(x=df_text['count'],
                            y=df_olap_norm_reduced.index.values.tolist(),
                            orientation='h',
                            marker={'color': 'grey'}))
    # Edit layout for multiplot
    for i in range(len(fig1.data)):
        fig1.data[i].xaxis='x1'
        fig1.data[i].yaxis='y1'
    fig1.layout.yaxis1.update({'anchor': 'x1'})
    fig1.layout.xaxis1.update({'anchor': 'y1', 'domain': [0, .75]})
    for i in range(len(fig2.data)):
        fig2.data[i].xaxis='x2'
        fig2.data[i].yaxis='y2'
    # Initialize xaxis2 and yaxis2
    fig2['layout']['xaxis2'] = {}
    fig2['layout']['yaxis2'] = {}
    fig2.layout.yaxis2.update({'anchor': 'x2', 'showticklabels': False})
    fig2.layout.xaxis2.update({'anchor': 'y2', 'domain': [.76, 1]})
    # Multi plot
    fig = go.Figure()
    fig.add_traces([fig1.data[0], fig2.data[0]])
    fig.layout.update(fig1.layout)
    fig.layout.update(fig2.layout)   
    return fig

def png_output(size):
    png_renderer = pio.renderers["png"]
    png_renderer.width = size[0]
    png_renderer.height = size[1]
    pio.renderers.default = "png"


########################################################################################
###                       Mining & Confirmatory Analysis functions                   ###
########################################################################################

def elbow_method(X, n_cluster_max):
    """"Within Cluster Sum of Squares (WCSS) method for optimal number of clusters identification"""
    wcss, sil = [], []  # Within Cluster Sum of Squares (WCSS) & silhouette index
    for i in range(2, n_cluster_max):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans_pred_y = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)  # WCSS
        sil.append(metrics.silhouette_score(X, kmeans_pred_y, metric="euclidean"))  # Silhouette score
    return wcss, sil

def similarity_index_plot(wcss, sil):
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].scatter(range(2,len(sil)+2),sil,
                   c='c', marker='v')
    axs[1].scatter(range(2,len(wcss)+2),wcss,
                   c='r', marker='o')
    # Legend
    axs[0].set_ylabel('Silhouette')
    axs[1].set_ylabel('Cluster Sum of Squares')
    axs[1].set_xlabel('cluster number')
    # Set ticks inside
    plt.xticks(range(2,len(sil)+2), range(2,len(sil)+2))
    axs[0].tick_params(axis="y", direction="in", left="off", labelleft="on")
    axs[0].tick_params(axis="x", direction="in", left="off", labelleft="on")
    axs[1].tick_params(axis="x", direction="in", left="off", labelleft="on")
    axs[1].tick_params(axis="y", direction="in", left="off", labelleft="on")
    axs[0].grid(axis='y', color='grey', linestyle='--', linewidth=0.5, alpha=0.4)
    axs[1].grid(axis='y', color='grey', linestyle='--', linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    plt.show()
    return fig


### Visualization

def cluster_counter_plot(counts, title=None):
    """Plot motif counts per cluster with whiskers"""
    stats = counts.describe()
    stats = stats.transpose().sort_values(by=['50%'], ascending=False).transpose()
    keys = counts.columns
    y_pos = np.arange(len(keys))
    yerr_pos = stats.loc['75%'].values - stats.loc['50%'].values
    yerr_neg = stats.loc['50%'].values - stats.loc['25%'].values
    # Plot
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    plt.bar(y_pos,
             stats.loc['50%'].values,
             yerr = [yerr_neg, yerr_pos],
             tick_label=keys,
             align='center',
             alpha=0.4)
    hfont = {'fontname': 'Times New Roman'}
    ax.tick_params(axis="y", direction="in", left="off", labelleft="on", labelsize=13)
    ax.tick_params(axis="x", direction="in", left="off", labelleft="on", labelsize=13)
    plt.xticks(y_pos, keys, rotation=90, **hfont)
    plt.yticks(**hfont)
    #plt.xlabel('Symbolic Aggregate Approximation sequences', **hfont)
    plt.ylabel('counts', fontsize=15, **hfont)
    plt.title(title, **hfont)
    plt.tight_layout()
    plt.show()
    return fig