#single page
#nice aesthetics
#post on github
# fitting histograms to statistical data

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
import pandas as pd
import io

#Title
st.title("Sebastian's Magical Statistical Distribution Fitter")
st.header('Create a histogram and fit it to common statistical distributions in seconds!')
st.text('')
st.text('Upload a CVS file or Manually Input your data to begin:')
st.divider()
#Data Entry
#Manual Data Entry
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'csv_buffer' not in st.session_state:
    st.session_state.csv_buffer = None
#file uploader
st.header('Upload')
uploaded_file = st.file_uploader("Upload your CSV fie", type=["csv"])
show_manual_input = uploaded_file is None
if show_manual_input:
    st.header('Manual Input')
    #manual input UI --> CSV memory file
    col_name = st.text_input('Enter the column title:')
    values_input = st.text_area("Enter numeric values seperated by commas:")
    if st.button("Add Column"):
        if col_name and values_input:
            values = [v.strip() for v in values_input.split(',')]
            #column padding
            df = st.session_state.df
            new_len = len(values)
            # Gather lengths of existing columns
            existing_lengths = [len(df[col]) for col in df.columns] if not df.empty else []
            max_len = max([new_len] + existing_lengths) if existing_lengths else new_len
            df = df.reindex(range(max_len))
            # Pad existing columns upward
            for col in df.columns:
                current_len = len(df[col])
                if current_len < max_len:
                    df[col] = df[col].tolist() + [''] * (max_len - current_len)
            # Pad the new column
            if new_len < max_len:
                values = values + [''] * (max_len - new_len)
            # Assign the new column
            df[col_name] = values
            st.session_state.df = df
            #save dataframe to in-memory CSV
            csv_buffer = io.StringIO()
            st.session_state.df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.session_state.csv_buffer = csv_buffer
        else:
            st.error("Please provide both a column name and values")
    #clear button
    if st.button('Clear Data'):
        if 'df' in st.session_state:
            del st.session_state.df
        if 'csv_buffer' in st.session_state:
            del st.session_state.csv_buffer
        st.rerun()
#Store ufile as CSV memory or Uploaded data
if uploaded_file is not None:
    ufile = io.StringIO(uploaded_file.getvalue().decode('utf-8'))
    ufile.seek(0)
    st.session_state.csv_buffer = ufile
else:
    if st.session_state.csv_buffer is None:
        st.session_state.csv_buffer = io.StringIO()
    ufile = st.session_state.csv_buffer

#Format Data
ufile.seek(0)
csv_content = ufile.getvalue().strip()

if not csv_content:
    st.write('Please Upload File!') 

else:
    st.write('File Uploaded Successfully!')
    st.divider()

    #Convert data frame into dictionary of 1D np.arrays and their respective titles 
    ufile_dataframe = pd.read_csv(ufile)
    ufile_numdata = ufile_dataframe.select_dtypes(include='number')
    ufile_numdata = ufile_numdata.loc[:, 
    (ufile_numdata != 0).any() & 
    (~ufile_numdata.isna().all()) & 
    (ufile_numdata.nunique() > 1)
]
    ufile_dict = {column_name: ufile_numdata[column_name].values for column_name in ufile_numdata.columns}

    #show initial data youre working with
    st.subheader('Chosen File')
    st.dataframe(ufile_numdata)
    st.divider()

    #accept user input: select what columns you want made into a histogram or click all
    st.subheader('Graphing Options')
    st.text('')
    opts = list(ufile_dict.keys())
    multiselect_placeholder = st.empty()
    allfunc = st.checkbox('Use All')
    if allfunc:
        option_select = opts
        multiselect_placeholder.empty()
    else:
        option_select = multiselect_placeholder.multiselect('Select Which Columns You Wish To Use', options = opts)
    #option_select: list of keys

    #allocate values corresponding to the desired columns
    usedval_list = list()
    for key in option_select:
        usedval = ufile_dict[key]
        usedval_list.append(usedval)
    #usedval_list: list of arrays

#Distribution Functions: accepts x array and ouputs dist heights at xi
    def norm(x, mu=None, sigma=None):
          fitted_mu,fitted_sigma = sci.norm.fit(x)
          mu = fitted_mu if mu is None else mu
          sigma = fitted_sigma if sigma is None else sigma
          #gives dist object: infomration about the normal distribution based on x
          dist = sci.norm(mu, sigma)
          #PDF Vals, numpy array of height of dist heights at each value x
          return dist, {'mu':mu, 'sigma':sigma}
    def exponential(x, loc=None, scale=None):
        fitted_loc, fitted_scale = sci.expon.fit(x)
        loc = fitted_loc if loc is None else loc
        scale = fitted_scale if scale is None else scale
        dist = sci.expon(loc, scale)
        return dist, {'loc':loc, 'scale':scale}
    def uniform(x, loc=None, scale=None):
        fitted_loc, fitted_scale = sci.uniform.fit(x)
        loc = fitted_loc if loc is None else loc
        scale = fitted_scale if scale is None else scale
        dist = sci.uniform(loc, scale)
        return dist, {'loc':loc, 'scale':scale}
    def skew_normal(x, a=None, loc=None, scale=None):
        fitted_a, fitted_loc, fitted_scale = sci.skewnorm.fit(x, floc=0) 
        a = fitted_a if a is None else a
        loc = fitted_loc if loc is None else loc
        scale = fitted_scale if scale is None else scale
        dist = sci.skewnorm(a, loc=loc, scale=scale)
        return dist, {'a': a, 'loc': loc, 'scale': scale}
    def laplace(x, loc=None, scale=None):
        fitted_loc, fitted_scale = sci.laplace.fit(x)
        loc = fitted_loc if loc is None else loc
        scale = fitted_scale if scale is None else scale
        dist = sci.laplace(loc, scale)
        return dist, {'loc':loc, 'scale':scale}
    def beta(x, a=None, b=None, loc=None, scale=None):
        fitted_a, fitted_b, fitted_loc, fitted_scale = sci.beta.fit(x)
        a = fitted_a if a is None else a
        b= fitted_b if b is None else b
        loc = fitted_loc if loc is None else loc
        scale = fitted_scale if scale is None else scale
        dist = sci.beta(a, b, loc=loc, scale=scale)
        return dist, {'a':a, 'b':b, 'loc':loc, 'scale': scale}
    def studentT(x, df=None, loc=None, scale=None):
        fitted_df, fitted_loc, fitted_scale = sci.t.fit(x)
        df = fitted_df if df is None else df
        loc = fitted_loc if loc is None else loc
        scale = fitted_scale if scale is None else scale
        dist = sci.t(df, loc=loc, scale=scale)
        return dist, {'df':df,'loc':loc,'scale':scale}
    def chi_squared(x, df=None, loc=None, scale=None):
        fitted_df, fitted_loc, fitted_scale = sci.chi2.fit(x)
        df = fitted_df if df is None else df
        loc = fitted_loc if loc is None else loc
        scale = fitted_scale if scale is None else scale
        dist = sci.chi2(df, loc=loc, scale=scale)
        return dist, {'df':df,'loc':loc,'scale':scale}
    def weibull(x, c=None, loc=None, scale=None):
        fitted_c, fitted_loc, fitted_scale = sci.weibull_min.fit(x)
        c = fitted_c if c is None else c
        loc = fitted_loc if loc is None else loc
        scale = fitted_scale if scale is None else scale
        dist = sci.weibull_min(c, loc=loc, scale=scale)
        return dist, {'c':c, 'loc':loc,'scale':scale}
    def cauchy(x, loc=None, scale=None):
        fitted_loc, fitted_scale = sci.cauchy.fit(x)
        loc = fitted_loc if loc is None else loc
        scale = fitted_scale if scale is None else scale
        dist = sci.laplace(loc=loc, scale=scale)
        return dist, {'loc':loc, 'scale':scale}
    
#Select Distribution Function
    distfunc_options_dict = {'Norm':norm,'Exponential':exponential,'Uniform':uniform,'Skew-Normal':skew_normal,'Laplace':laplace,'Beta':beta,"Student's T":studentT,'Chi_Squared':chi_squared,'Weibull':weibull,'Cauchy':cauchy}
    distfunc_options = list(distfunc_options_dict.keys())
    distfunc_selected = st.selectbox('Select Which Distribution You Wish To Apply', options= distfunc_options)
    st.divider()
    #distfunc_selected = selected key from distfunc_options

#Make the histogram w/ distribution data based on the data
    for i, arr in enumerate(usedval_list):
        #i is the array currently iterated over, arr is the array
        st.header(str(option_select[i]))
        #calculate numer of bins using Freedman_Diaconis method
        numrows = len(arr)
        #filter NaN objects
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        iqr = np.percentile(arr, 75) - np.percentile(arr,25)
        if iqr == 0:
            binnum =1
        else: 
            binnum =  binnum = int(np.ceil((arr.max() - arr.min()) * numrows**(1/3) / (2 * iqr)))
        #plot histogram
        fig, ax = plt.subplots()
        counts, bins, patches = ax.hist(arr, bins=binnum, edgecolor='black', label='Selected Data')
        #lock histogram scale
        xmin,xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax)
        ymin,ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        #fit dist and get parameter
        bin_width = bins[1]-bins[0]
        dist_func = distfunc_options_dict[distfunc_selected]
        dist_obj, params = dist_func(arr)
        #safe streamlit slide option w/ min and max val buffers
        slider_params={}
        for param_name, val in params.items():
            min_val = float(min(arr))
            max_val = float(max(arr))
            #buffer
            if min_val == max_val:
                min_val -=0.1
                max_val +=0.1
            #labels
            pretty_labels = {
            'mu': 'Mean (μ)',
            'sigma': 'Standard Deviation (σ)',
            'loc': 'Location',
            'scale': 'Scale',
            'a': 'Alpha',
            'b': 'Beta',
            'df': 'Degrees of Freedom',
            'c': 'Shape',
            'shape':'Shape'
            }
            label = pretty_labels.get(param_name, param_name)
            #make slider
            slider_params[param_name] = st.slider(label, min_value=min_val, max_value=max_val, value=float(val))

        # Recompute distribution with user
        dist_obj, params_updated = dist_func(arr, **slider_params)

        #scale pdf_values
        x = np.linspace(arr.min(), arr.max(), len(arr))
        pdf_values = dist_obj.pdf(x)
        scaled_pdf = pdf_values * len(arr) *bin_width
        #overlay plot: ax.plot(blahblahblah)
        ax.plot(x,scaled_pdf, label='Fitted Statistical Distribution')
        #labels
        ax.set_title(str(distfunc_selected)+' Fitted Graph of '+str(option_select[i]))
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.legend() 
        #Show visuals
        st.pyplot(fig)
        #midpoint error calc
        bin_centers = 0.5*(bins[:-1]+bins[1:])
        pdf_at_bins = np.interp(bin_centers, x, scaled_pdf)
        vertical_distances = np.abs(counts-pdf_at_bins)
        try:
            raw_max = max(float(counts.max()),float(pdf_at_bins.max()))
        except ValueError:
            raw_max = 0.0
        max_value = max(raw_max,1e-12)
        normalized_errors = vertical_distances/max_value
        if not np.isfinite(normalized_errors).all():
            shape_accuracy = float('inf')
        else:
            shape_accuracy = np.mean(normalized_errors)*100 
        # Find area of histogram and Stat Fit
        hist_area = np.sum(counts*bin_width)
        #Calculate overlap area
        overlap_area = 0
        for i in range(len(counts)):
            #find rectangle midpoint
            bin_start = bins[i]
            bin_end = bins[i+1]
            bin_mid = 0.5*(bin_start+bin_end)
            #fit continous stat dist to each midpoint
            stat_val = np.interp(bin_mid, x, scaled_pdf)
            #approximate overlap area
            overlap_area += min(counts[i], stat_val)* bin_width
            #Calculate percent diff
        o_percent = round((overlap_area/hist_area)*100)

        #Write data
        st.write('Fit Quality:')
        if shape_accuracy != float('inf'):
            overlap_fraction = max(o_percent/100,1e-12)
            adjusted_faq = shape_accuracy/overlap_fraction
            st.write('- Fit Accuracy Quotient (FAQ): '+str(round(adjusted_faq)))
            st.caption('> The closer the FAQ of a data set is to 0, the better the fit')
            st.write("- Fitted Curve Covers ~"+str(o_percent)+'% Of Selected Dataset')
        else:
            st.write('- Fit Accuracy Quotient (FAQ): N/A')
            st.write('- Overlap N/A')
        st.divider()


    




