# KOC+ or OCKELM+
One-class KRR or One-class KELM with Privileged Information

If you are using this code then kindly cite the folowing paper:

OCKELM+: Kernel Extreme Learning Machine based One-class Classification using Privileged Information (or KOC+: Kernel Ridge Regression or  Least Square SVM with zero bias based One-class Classification using Privileged Information)

Author: C. Gautam, A. Tiwari, M. Tanveer

#(Paper is under review in Information Sciences)

# For reproducing the results of Heart datasets:

--  Open KOC+_Heart_Experiments.ipynb in Python notebook and run all cells. It will save all results in .pkl files. Results on optimal   parameters along with optimal parameters values will be saved in a excel file.   

--  Be dfault these codes produce results for group attribute 'Age'. For other two group attributes (Sex and Electrocardiographic): change the value in cell number 3 and 4 as follows:

For group attribute = Sex:

Uncomment this line in cell 3:  
 privileged_space = privileged_space_tot.ix[:]['p1']

Uncomment this line in cell 4:
 feature_space = feature_space.drop('a2', axis=1)
 privfeat = 'Sex'

For group attribute = Electrocardiographic:

Uncomment this line in cell 3:  
 privileged_space = privileged_space_tot.ix[:]['p3']

Uncomment this line in cell 4:
 feature_space = feature_space.drop('a7', axis=1)
 privfeat = 'Elect'
