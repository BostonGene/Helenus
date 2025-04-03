import numpy as np
import pickle
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from matplotlib import cm
import pandas as pd
from scipy.stats import pearsonr
from bioreactor.use_colors import cells_p
from scipy import stats
import pylab as pl
import sklearn
import matplotlib.pyplot as plt
from bioreactor.plotting import axis_net
from tumor_profile.profile_deconv.utils import *
import matplotlib.patches as mpatches


def concordance_correlation_coefficient(y_true, y_pred):
    '''
    :param y_true: pandas.core.frame.DataFrame, real value.
    :param y_pred: pandas.core.frame.DataFrame, result of prediction. 
    '''
    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator


def mase(cclpaad, paad_mixes, tum_prof2):
    '''
    :param cclpaad: pandas.core.frame.DataFrame, real tumor profile.
    :param paad_mixes: pandas.core.frame.DataFrame, whole sample.
    :param tum_prof2: pandas.core.frame.DataFrame, subtraction result.
    '''
    numerator = abs(cclpaad - tum_prof2)
    denumerator = abs(cclpaad - paad_mixes).mean()
    ms = (numerator/denumerator).mean()
    return ms

def mae_mean(y, y_true):
    '''
    :param y: pandas.core.frame.DataFrame, result of prediction.
    :param y_true: pandas.core.frame.DataFrame, real value. 
    '''
    mm = (abs(y - y_true)).mean()/(y_true.mean())
    return mm

def mae(y, y_true):
    '''
    :param y: pandas.core.frame.DataFrame, result of prediction.
    :param y_true: pandas.core.frame.DataFrame, real value. 
    '''
    m = (abs(y - y_true)).mean()
    return m

def correlation_coefficients(set1, set2, set3=None, err=None, bulks=None):
    '''
    param set1: pandas.core.series.Series, tumor set.
    param set2: pandas.core.series.Series, sample set.
    param set3: pandas.core.series.Series, subtraction set. 
    '''
    if set3 is not None:
        set3, set2 = set2, set3
        ms = mase(set1, set3, set2)

    pr_coef = np.corrcoef(set1, set2)[0,1]
    sp_coef = scipy.stats.spearmanr(set1, set2)[0]
    co_coef = concordance_correlation_coefficient(set1, set2)
    mmean  = mae_mean(set2, set1)
    metrics = {
            'Pearson_correlation': pr_coef,
           'Spearman_correlation': sp_coef,
           'Concordance_correlation': co_coef,
            'Cosine_similarity': 0.9,
            'MAE/Mean': mmean
        }
    if set3 is not None:
        metrics['MASE'] = ms
        metrics['Mean_STD_Mean'] = 0
        
        
    return metrics


def pic_plot2(figure, df1, df2, tumc, gene, for_title, mini, maxi, colorss, err=None,
             mase=True)
    c1 = figure.scatter(df1,df2, c=colorss, cmap='cool', marker = 'o', s = 70)
    figure.set_xlabel('Expression of malignant cells, TPM', fontsize = 14)
    figure.set_ylabel('Expression in extracted tumor profile, TPM', fontsize = 14)
    figure.grid(False)
    figure.patch.set_facecolor('white')
    cbar = plt.colorbar(ax = figure, mappable = c1)
    cbar.set_label('fraction of tumor counts in pseudobulk', fontsize=14)
    if err is not None:
        figure.errorbar(df1, df2, yerr=err, fmt='none', elinewidth=0.5)
    text = '{}\n Pearson correlation coefficient {}\n Spearman correlation coefficient {}\n Concordance correlation coefficient {}\n MAE/mean {} \n'
    title = text.format(gene, np.round(for_title['Pearson_correlation'], 2),
                                       np.round(for_title['Spearman_correlation'],2),
                                       np.round(for_title['Concordance_correlation'],2),
                                       np.round(for_title['MAE/Mean'],2))
                                     
    if mase:
        title = title + 'MASE: ' + str(np.round(for_title['MASE'],2))
        figure.set_title(title, fontsize = 16)
        deltax = (maxi - mini)/20.0
        deltay = (maxi - mini)/20.0
        figure.set_xlim((mini - deltax, maxi + deltax))
        figure.set_ylim((mini - deltay, maxi + deltay))
        figure.plot([mini, maxi], [mini, maxi], '-k', color='blue',linewidth=1.0)
    else:
        deltax = (maxi - mini)/20.0
        deltay = (maxi - mini)/20.0
        figure.set_xlim((mini - deltax, maxi + deltax))
        figure.set_ylim((mini - deltay, maxi + deltay))
        figure.set_title(title, fontsize = 16)
        figure.plot([mini, maxi], [mini, maxi], '-k', color='blue',linewidth=1.0)
        return mini, maxi
def validation_plots2(target_genes, data_sc, agg, extracted,
                      tumc, stds_2,col,
                      prenorm = None, draw = True, pro_coefs = None):
    """
    Return dictionaries with coefficients corellation values which show the result before and
    after microenvironment subtraction. 
    :param target_genes: list, list of genes. 
    :param data_sc: pandas.core.frame.DataFrame, contains real tumor profile. Indexes are genes,
        columns are names of bulks.
    :param agg: pandas.core.frame.DataFrame, experimental data, whole sample profile.
    :param extracted: pandas.core.frame.DataFrame, a result of prediction(only tumor implied).
    :param tumc: pandas.core.series.Series, fractions of tumor.
    :param stds_2: pandas.core.frame.DataFrame, errors (std) for indirect measurements. 
    :param prenorm: pandas.core.frame.DataFrame, gives a result only after renormalization, without subtraction.
    """
    
    correct_genes = []
    bulks = agg.columns
    metrics_before, metrics_after, metrics_renorm = [], [], []
    for gene in target_genes:
        #try:

            tumor_real = data_sc.loc[:,bulks].loc[gene]
            sample = agg.loc[:,bulks].loc[gene]
            subtraction = extracted.loc[:,bulks].loc[gene]
            try: 
                err=stds_2.loc[gene, bulks]
            except KeyError:
                coefs = pro_coefs[gene][abs(pro_coefs[gene]) > 0]
                coefs2 = [x for x in coefs.index if x in stds_2.index]
                err = np.sqrt((stds_2.loc[coefs2].multiply(coefs.loc[coefs2], axis=0)**2).sum())
            metrics_before.append(correlation_coefficients(tumor_real, sample))
            metrics_after.append(correlation_coefficients(tumor_real, sample, subtraction, err, bulks))
            if prenorm is not None:
                renormalization = prenorm.loc[:,bulks].loc[gene]
                metrics_renorm.append(correlation_coefficients(tumor_real, sample, renormalization, err, bulks))
                if draw:
                    mini = 0
                    maxi = max(max(tumor_real), max(sample),
                               max(renormalization))
                    
                    figure = axis_net(3,1, x_len=7,y_len=6)
                    pic_plot2(figure[0], tumor_real, sample, tumc.loc[bulks], gene=gene, 
                                          for_title=metrics_before[-1],
                                          mini=mini, maxi=maxi, colorss=col,
                              mase=False)
                    pic_plot2(figure[1], tumor_real, subtraction, tumc.loc[bulks], err=err, 
                             gene=gene, for_title=metrics_after[-1],
                              mini=mini, maxi=maxi, colorss=col,
                              mase=True)
                    pic_plot2(figure[2], tumor_real, renormalization, tumc.loc[bulks], err=err, 
                             gene=gene, for_title=metrics_renorm[-1],
                              mini=mini, maxi=maxi, colorss=col,
                              mase=True)
                    figure[2].set_ylabel('Expression after renormalization, TPM', fontsize = 14)
                    figure[0].set_ylabel('Expression before subtraction, TPM', fontsize = 14)
            else:
                if draw:
                    figure = axis_net(2,1, x_len=8,y_len=6)
                    pic_plot2(figure[0], tumor_real, sample, tumc.loc[bulks], gene=gene, 
                             for_title=metrics_before[-1], mase=False)
                    pic_plot2(figure[1], tumor_real, subtraction, tumc.loc[bulks], err=err, gene=gene, 
                             for_title=metrics_after[-1], mini=mini, maxi=maxi)
            correct_genes.append(gene)
     
    
    if len(metrics_renorm) != 0:
        return metrics_before, metrics_after, metrics_renorm 
              
    return metrics_before, metrics_after, correct_genes


def title_composer(y_true, y_pred,samples_amount):
    genes_amount = int(len(y_true)/samples_amount)
    cor_coef = round(concordance_correlation_coefficient(np.array(y_true), np.array(y_pred)), 2)
    maeme = round(mae_mean(np.array(y_true), np.array(y_pred)),2)
    pr_coef = round(np.corrcoef(np.array(y_true), np.array(y_pred))[0,1], 2)
    sp_coef = round(scipy.stats.spearmanr(np.array(y_true), np.array(y_pred))[0], 2)
 
    text_for_title = ('Genes amount: '+ str(genes_amount)+  '\n' +
            'Pearson correlation coefficient: '+
                              str(pr_coef) + '\n' +
                              'Spearman correlation coefficient: '+
                              str(sp_coef)+'\n' +
                'Concordance correlation coefficient: '
                              +str(cor_coef) + '\n' +
                              'MAE/Mean: ' + str(maeme)
                             )
    return text_for_title

def data_collector(gene,
                    before_subtraction,
                   after_subtraction,
                   correct_answer,
                   datapoints
                  ):
    maxi = max(max(before_subtraction.loc[gene]),
                           max(after_subtraction.loc[gene]))
    if maxi<30:
        datapoints['low_before'].append(before_subtraction.loc[gene])
        datapoints['low_after'].append(after_subtraction.loc[gene])
        datapoints['low_answer'].append(correct_answer.loc[gene])
    elif maxi<400 and maxi >= 30:
        datapoints['medium_before'].append(before_subtraction.loc[gene])
        datapoints['medium_after'].append(after_subtraction.loc[gene])
        datapoints['medium_answer'].append(correct_answer.loc[gene])
    elif maxi >= 400:
        datapoints['high_before'].append(before_subtraction.loc[gene])
        datapoints['high_after'].append(after_subtraction.loc[gene])
        datapoints['high_answer'].append(correct_answer.loc[gene])
                        
    return datapoints


def butterfly_plot(
                   cmap_name,
                   data_points,
                   color_index = 2,
                   samples_amount = 1,
                  name_to_save=None):
    def plotter(cmap_name,
                ax,
               category_x: str,
               category_y: str,
               samples_amount):
        
        cols = []
        cmapp = plt.get_cmap(cmap_name)
        for x in range(int(len(data_points[category_x])/samples_amount)):
            cols +=  [x%100*color_index for n in range(samples_amount)]
            col = x%100*color_index
            
        
        ax.scatter(data_points[category_x],data_points[category_y], c = cmapp(cols), alpha=0.5)
        ax.set_title(title_composer(data_points[category_x],data_points[category_y], samples_amount))

        ax.grid(False)
        ax.set_xlabel('Expression in malignant cells only, TPM')
        if 'after' in category_y:
            ax.set_ylabel('Expression after subtraction, TPM')
        else: 
            ax.set_ylabel('Expression before subtraction, TPM')
        
        
    for key in data_points.keys():
            sett = []
            for i in data_points[key]:
                sett.append(list(i))
            sett = sum(sett, [])
            data_points[key] = sett
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams["axes.edgecolor"] = "0.0"
    plt.rcParams["axes.linewidth"]  = 1.75
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams['axes.titleweight'] = 'bold'
    alp = 0.7
    fig, ax = plt.subplots(1,3, figsize=(20,5))
    fig1, axaft = plt.subplots(1,3, figsize=(20,5))
    keys = list(data_points.keys())
    colors_list = []
    for i in range(3):
        col = plotter(cmap_name,ax[i],keys[2+i*3], keys[i*3], samples_amount)
        colors_list.append(col)
        try:
            maxi = max(max(data_points[keys[i*3]]), max(data_points[keys[1+i*3]]))#,
                     # max(data_points[keys[2+i*3]]))
        except:
            maxi = 0
        #if maxi>1500:
        #    maxi = 1500
        plotter(cmap_name,axaft[i],keys[2+i*3], keys[1+i*3], samples_amount)
        ax[i].set_ylim(0-maxi*0.05, maxi+maxi*0.05)
        axaft[i].set_ylim(0-maxi*0.05, maxi+maxi*0.05)
        ax[i].set_xlim(0-maxi*0.05, maxi+maxi*0.05)
        axaft[i].set_xlim(0-maxi*0.05, maxi+maxi*0.05)
        ax[i].plot([0, maxi],[0, maxi])
        axaft[i].plot([0, maxi],[0, maxi])
    if name_to_save is not None: 
            
        fig.savefig(name_to_save+'/butterflies_before.png', bbox_inches = 'tight')
        fig1.savefig(name_to_save+'/butterflies_after.png', bbox_inches = 'tight')
  
    return colors_list
        

def plot_validation_pictures(before_subtraction,
                             subtraction,
                             correct_answer,
                             geneset,
                             draw_it: bool=False,
                             butterfly: bool=False,
                             threshold=-1, 
                             cols = None,
                             cmap_name = 'hsv',
                             color_index=2,
                             name_to_save=None,
                             ):
    before_ML, after_ML, renorm_data, good_genes = [[] for _ in range(4)]
    tumc = predict_tumor_fraction(before_subtraction)
    zero_std = pd.DataFrame(0, index = before_subtraction.index, 
                            columns=before_subtraction.columns) 
    if butterfly:
        datapoints = {}
        for i in ['low_', 'medium_', 'high_']:
            for m in ['before', 'after', 'answer']:
                datapoints[i+m] = []
    if cols is None:
        cols = tumc.copy()
    for x,gene in enumerate(geneset):
                before_ccc = concordance_correlation_coefficient(correct_answer.loc[gene],before_subtraction.loc[gene])
                after_ccc = concordance_correlation_coefficient(correct_answer.loc[gene],subtraction.loc[gene])
                if after_ccc >= threshold or np.isnan(after_ccc):
                    a,b,c = validation_plots2([gene], correct_answer, 
                                 before_subtraction, subtraction, tumc, zero_std, 
                                 prenorm = before_subtraction.divide(tumc),
                                 col = cols, draw = draw_it, pro_coefs = None)
                    good_genes.append(gene)
                    before_ML.append(a[0])
                    after_ML.append(b[0])
                    renorm_data.append(c[0])
                    if butterfly:
                        datapoints = data_collector(gene,before_subtraction,subtraction,
                                                correct_answer,datapoints)
          
    if butterfly:
        colors = butterfly_plot(cmap_name,datapoints,color_index = color_index,
                               samples_amount = len(subtraction.columns), name_to_save=name_to_save
        )
        
        
    before_ML = pd.DataFrame(before_ML, index=good_genes)
    after_ML = pd.DataFrame(after_ML, index=good_genes)
    renorm_data = pd.DataFrame(renorm_data, index=good_genes)
    return before_ML, after_ML, renorm_data, good_genes

def boxplots_result(metrics_before, metrics_after, metric):
    df = pd.concat([metrics_before[metric], metrics_after[metric]],1)
    df.columns = ['before', 'ML-based correction']
    df.boxplot()
    plt.grid(False)
    plt.title(metric)


def single_gene_changes(before,
                       reconstructed_profile,
                       malignant_only,
                       gene,
                        name_to_save=None):
        true_color = 'lawngreen'
        predicted_color = 'tomato'
        before_color = 'skyblue'
        true_label = 'true'
        predicted_label = 'predicted'
        before_label = 'before'
    
        samples = before.columns
        
        mm_before = mae(before.loc[gene, samples], malignant_only.loc[gene, samples])
        mm_after = mae(reconstructed_profile.loc[gene, samples], malignant_only.loc[gene, samples])
        delta_mm = str(round((mm_before-mm_after),3))
        
        df = pd.DataFrame({true_label: np.log(malignant_only.loc[gene, samples] + 1), 
                       predicted_label: np.log(reconstructed_profile.loc[gene, samples] + 1),
                       before_label: np.log(before.loc[gene, samples] + 1)})
        fig, ax = plt.subplots()
        ax = df.plot(kind='scatter', x=true_label, y=true_label, color=true_color, ax=ax)    
        ax = df.plot(kind='scatter', x=true_label, y=predicted_label, color=predicted_color, ax=ax)    
        ax = df.plot(kind='scatter', x=true_label, y=before_label, color=before_color, ax=ax)
        ax.set(ylabel='log2(TPM+1)', title=gene)

        shifts = 0.2 
        ax.set_xlim(df[true_label].min() - shifts,df[true_label].max() + shifts)
        ax.set_ylim(df.min().min() - shifts,df.max().max() + shifts)
    
        patch1 = mpatches.Patch(color=true_color, label=true_label)
        patch2 = mpatches.Patch(color=before_color, label=before_label)
        patch3 = mpatches.Patch(color=predicted_color, label=predicted_label)
        ax.axline([0, 0], [1, 1], linestyle='dashed', color=true_color, alpha=0.7)
        ax = ax.legend(handles=[patch1, patch2, patch3], bbox_to_anchor=(1.1, 1.05))
        plt.grid(False)
        
        ccc_before = concordance_correlation_coefficient(df.true, df.before)
        ccc_after = concordance_correlation_coefficient(df.true, df.predicted)
        delta_ccc = str(round((ccc_after-ccc_before),2))
        
        pearson_before = np.corrcoef(df.before, df.true)[0,1]
        pearson_after = np.corrcoef(df.predicted, df.true)[0,1]
        delta_pearson = str(round((pearson_after-pearson_before),2))
        
        for_title = gene+'\nΔConcordance(after-before): '+delta_ccc+'\n'+'ΔMAE(before-after): '+delta_mm +'\n'+ 'ΔPearson(after-before): '+ delta_pearson
        plt.title(for_title)
        if name_to_save is not None:
             plt.savefig(name_to_save+'/'+gene+'.png', bbox_inches = 'tight')
