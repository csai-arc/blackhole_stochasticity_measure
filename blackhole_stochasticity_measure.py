from TIRE import DenseTIRE as TIRE
import torch
import numpy as np
import matplotlib.pyplot as plt
from TIRE import utils
from scipy import integrate
from scipy.fft import fft
import TIRE.simulate as simulate
from scipy.stats import entropy
import pandas as pd
import csv

txtfilename_train_list=['pink_noise.txt','white_noise.txt','x00.085000.txt','x00.160000.txt','x00.235000.txt','x00.310000.txt','x00.385000.txt','x00.460000.txt','x00.535000.txt','x00.610000.txt','x00.685000.txt','x00.760000.txt','x00.835000.txt','x00.910000.txt']

ts3=np.empty(0)
for idx in range(len(txtfilename_train_list)):
    txtfilename2='./train_data/'+txtfilename_train_list[idx]
    ts2=np.loadtxt(txtfilename2)
    #std=np.std(ts2)
    
    minimum=(min(ts2))
    ts2=ts2-minimum
    maximum=(max(ts2))
    ts2=ts2/maximum

    ts3=np.append(ts3,ts2)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim = 1

model = TIRE(dim,window_size=10,intermediate_dim_TD=10,intermediate_dim_FD=10).to(device)
model.fit(np.array(ts3), epoches=200)

txtfilename_list=['sac_ascf_alpha','sac_ascf_beta','sac_ascf_delta','sac_ascf_gamma','sac_ascf_kai','sac_ascf_kappa','sac_ascf_lambda','sac_ascf_mu','sac_ascf_nu','sac_ascf_phi','sac_ascf_rho','sac_ascf_theta']
txtfilename_list_2=['lordata','lorenz']
txtfilename_train_list_noise=['pink_noise.txt','white_noise.txt']
txtfilename_train_list_new=['x00.085000.txt','x00.160000.txt','x00.235000.txt','x00.310000.txt','x00.385000.txt','x00.460000.txt','x00.535000.txt','x00.610000.txt','x00.685000.txt','x00.760000.txt','x00.835000.txt','x00.910000.txt']

row_names=['dxws']
row_names=row_names+txtfilename_list+txtfilename_list_2+txtfilename_train_list_noise+txtfilename_train_list_new

outputs_path='./<output_folder_path>/'

###CV1 -- KL divergence between biased dissimilarity and normalized input signal ###


with open(outputs_path+"klvsbws_new_variable.csv", 'w') as f: 
    write = csv.writer(f) 
    write.writerow(row_names) 

    for dx in range(5,500,5): # 5 to 500
        print("window size",dx)

        #row_no=1
        #col_no=ws/5
        data_line=[]
        data_line.append(dx)
        # main signals
        for idx in range(len(txtfilename_list)):

            input_name=txtfilename_list[idx]
            txtfilename1='./time_series_all/'+input_name
            ts1=np.loadtxt(txtfilename1)
            if (dx<len(ts1)):
                minimum=(min(ts1))
                ts1=ts1-minimum
                maximum=(max(ts1))
                ts1=ts1/maximum

                dissimilarities, change_point_scores = model.predict(ts1)

                dissimilarities_biased=np.copy(dissimilarities)
                diff=len(ts1)-len(dissimilarities)
                clipped_ts1=ts1[round(diff/2):len(dissimilarities)+round(diff/2)]
                for r in range(0,len(clipped_ts1),dx):
                    int_bias=np.mean(clipped_ts1[r:r+dx])
                    dissimilarities_biased[r:r+dx]=dissimilarities_biased[r:r+dx]+int_bias

                kl_score1=entropy(clipped_ts1,dissimilarities_biased)
            else:
                kl_score1=None

            #print(input_name,kl_score1)
            #row_no=row_no+idx
            data_line.append(kl_score1)


        # lor signals
        for idx in range(len(txtfilename_list_2)):
            

            input_name=txtfilename_list_2[idx]
            txtfilename1='./time_series_all/'+input_name
            ts1=np.loadtxt(txtfilename1)
            if (dx<len(ts1)):
                minimum=(min(ts1))
                ts1=ts1-minimum
                maximum=(max(ts1))
                ts1=ts1/maximum

                dissimilarities, change_point_scores = model.predict(ts1)

                dissimilarities_biased=np.copy(dissimilarities)
                diff=len(ts1)-len(dissimilarities)
                clipped_ts1=ts1[round(diff/2):len(dissimilarities)+round(diff/2)]
                for r in range(0,len(clipped_ts1),dx):
                    int_bias=np.mean(clipped_ts1[r:r+dx])
                    dissimilarities_biased[r:r+dx]=dissimilarities_biased[r:r+dx]+int_bias

                kl_score1=entropy(clipped_ts1,dissimilarities_biased)
            else:
                kl_score1=None
            #print(input_name,kl_score1)
            #row_no=row_no+idx
            data_line.append(kl_score1)

        # synthetic noise signals
        for idx in range(len(txtfilename_train_list_noise)):

            input_name=txtfilename_train_list_noise[idx]
            txtfilename1='./noise/'+input_name
            ts1=np.loadtxt(txtfilename1)
            if (dx<len(ts1)):
                minimum=(min(ts1))
                ts1=ts1-minimum
                maximum=(max(ts1))
                ts1=ts1/maximum

                dissimilarities, change_point_scores = model.predict(ts1)

                dissimilarities_biased=np.copy(dissimilarities)
                diff=len(ts1)-len(dissimilarities)
                clipped_ts1=ts1[round(diff/2):len(dissimilarities)+round(diff/2)]
                for r in range(0,len(clipped_ts1),dx):
                    int_bias=np.mean(clipped_ts1[r:r+dx])
                    dissimilarities_biased[r:r+dx]=dissimilarities_biased[r:r+dx]+int_bias

                kl_score1=entropy(clipped_ts1,dissimilarities_biased)
            else:
                kl_score1=None
            #print(input_name,kl_score1)
            #row_no=row_no+idx
            data_line.append(kl_score1)

        # synthetic lambda signals
        for idx in range(len(txtfilename_train_list_new)):
            input_name=txtfilename_train_list_new[idx]
            txtfilename1='./non_stochastic_linear_map_stoch/'+input_name
            ts1=np.loadtxt(txtfilename1)
            if (dx<len(ts1)):
                minimum=(min(ts1))
                ts1=ts1-minimum
                maximum=(max(ts1))
                ts1=ts1/maximum

                dissimilarities, change_point_scores = model.predict(ts1)

                dissimilarities_biased=np.copy(dissimilarities)
                diff=len(ts1)-len(dissimilarities)
                clipped_ts1=ts1[round(diff/2):len(dissimilarities)+round(diff/2)]
                for r in range(0,len(clipped_ts1),dx):
                    int_bias=np.mean(clipped_ts1[r:r+dx])
                    dissimilarities_biased[r:r+dx]=dissimilarities_biased[r:r+dx]+int_bias

                kl_score1=entropy(clipped_ts1,dissimilarities_biased)
            else:
                kl_score1=None
            #print(input_name,kl_score1)
            #row_no=row_no+idx
            data_line.append(kl_score1)
        write.writerow(data_line) 

## coeffecient of variations are to calculated offline

###CV2 -- Coefficient of variation of the coeffecient of variations computed for various window sizes on prominence of peaks curve  ###

with pd.ExcelWriter(outputs_path+"covvsbws_variable.xlsx") as f:
    df = pd.DataFrame(columns=row_names)

    data_line=[]
    start_dx=200
    initial_offset=0
    
    # main signals
    for idx in range(len(txtfilename_list)):

        input_name=txtfilename_list[idx]
        txtfilename1='./time_series_all/'+input_name
        ts1=np.loadtxt(txtfilename1)

        minimum=(min(ts1))
        ts1=ts1-minimum
        maximum=(max(ts1))
        ts1=ts1/maximum

        dissimilarities, change_point_scores = model.predict(ts1)
        
        cps_len=len(change_point_scores)
        change_point_scores_full=np.copy(change_point_scores[initial_offset:cps_len-initial_offset])

        cov_list=[]
        cov_mean_list=[]
        for dx in range(start_dx,len(change_point_scores_full),5):
            #print(input_name,dx)
            rem=len(change_point_scores_full)%dx
            change_point_scores_new=change_point_scores_full[0:len(change_point_scores_full)-rem]
            cov_mean=[]
            for r in range(0,len(change_point_scores_new),dx):
                arr=change_point_scores_new[r:r+dx]
                cov=variation(arr, axis = 0)
                cov_mean.append(cov)
                cov_list.append(cov)
            cov_mean_list.append(np.mean(cov_mean))
        cov_final=variation(np.array(cov_list), axis = 0)*100
        cov_mean_final=variation(np.array(cov_mean_list), axis = 0)
        data_line.append(cov_final)
        print(input_name,cov_final,cov_mean_final)
        #print(input_name,min(cov_list),max(cov_list),cov_list[-1])


        # lor signals
    for idx in range(len(txtfilename_list_2)):

        input_name=txtfilename_list_2[idx]
        txtfilename1='./time_series_all/'+input_name
        ts1=np.loadtxt(txtfilename1)

        minimum=(min(ts1))
        ts1=ts1-minimum
        maximum=(max(ts1))
        ts1=ts1/maximum

        dissimilarities, change_point_scores = model.predict(ts1)

        cps_len=len(change_point_scores)
        change_point_scores_full=np.copy(change_point_scores[initial_offset:cps_len-initial_offset])

        cov_list=[]
        cov_mean_list=[]
        for dx in range(start_dx,len(change_point_scores_full),5):
            #print(input_name,dx)
            rem=len(change_point_scores_full)%dx
            change_point_scores_new=change_point_scores_full[0:len(change_point_scores_full)-rem]
            cov_mean=[]
            for r in range(0,len(change_point_scores_new),dx):
                arr=change_point_scores_new[r:r+dx]
                cov=variation(arr, axis = 0)
                cov_mean.append(cov)
                cov_list.append(cov)
            cov_mean_list.append(np.mean(cov_mean))
        cov_final=variation(np.array(cov_list), axis = 0)*100
        cov_mean_final=variation(np.array(cov_mean_list), axis = 0)
        data_line.append(cov_final)
        print(input_name,cov_final,cov_mean_final)
        #print(input_name,min(cov_list),max(cov_list),cov_list[-1])
        

        # synthetic noise signals
    for idx in range(len(txtfilename_train_list_noise)):

        input_name=txtfilename_train_list_noise[idx]
        txtfilename1='./noise/'+input_name
        ts1=np.loadtxt(txtfilename1)

        minimum=(min(ts1))
        ts1=ts1-minimum
        maximum=(max(ts1))
        ts1=ts1/maximum

        dissimilarities, change_point_scores = model.predict(ts1)

        cps_len=len(change_point_scores)
        change_point_scores_full=np.copy(change_point_scores[initial_offset:cps_len-initial_offset])

        cov_list=[]
        cov_mean_list=[]
        for dx in range(start_dx,len(change_point_scores_full),5):
            #print(input_name,dx)
            rem=len(change_point_scores_full)%dx
            change_point_scores_new=change_point_scores_full[0:len(change_point_scores_full)-rem]
            cov_mean=[]
            for r in range(0,len(change_point_scores_new),dx):
                arr=change_point_scores_new[r:r+dx]
                cov=variation(arr, axis = 0)
                cov_mean.append(cov)
                cov_list.append(cov)
            cov_mean_list.append(np.mean(cov_mean))
        cov_final=variation(np.array(cov_list), axis = 0)*100
        cov_mean_final=variation(np.array(cov_mean_list), axis = 0)
        data_line.append(cov_final)
        print(input_name,cov_final,cov_mean_final)
        #print(input_name,min(cov_list),max(cov_list),cov_list[-1])
        
        

        # synthetic lambda signals
    for idx in range(len(txtfilename_train_list_new)):

        input_name=txtfilename_train_list_new[idx]
        txtfilename1='./non_stochastic_linear_map_stoch/'+input_name
        ts1=np.loadtxt(txtfilename1)

        minimum=(min(ts1))
        ts1=ts1-minimum
        maximum=(max(ts1))
        ts1=ts1/maximum

        dissimilarities, change_point_scores = model.predict(ts1)

        cps_len=len(change_point_scores)
        change_point_scores_full=np.copy(change_point_scores[initial_offset:cps_len-initial_offset])

        cov_list=[]
        cov_mean_list=[]
        for dx in range(start_dx,len(change_point_scores_full),5):
            #print(input_name,dx)
            rem=len(change_point_scores_full)%dx
            change_point_scores_new=change_point_scores_full[0:len(change_point_scores_full)-rem]
            cov_mean=[]
            for r in range(0,len(change_point_scores_new),dx):
                arr=change_point_scores_new[r:r+dx]
                cov=variation(arr, axis = 0)
                cov_mean.append(cov)
                cov_list.append(cov)
            cov_mean_list.append(np.mean(cov_mean))
        cov_final=variation(np.array(cov_list), axis = 0)*100
        cov_mean_final=variation(np.array(cov_mean_list), axis = 0)
        data_line.append(cov_final)
        print(input_name,cov_final,cov_mean_final)
        #print(input_name,min(cov_list),max(cov_list),cov_list[-1])
        

    df.loc[1]=data_line#*100
    df.to_excel(f,sheet_name="Window_cov_changepoint_scores")





