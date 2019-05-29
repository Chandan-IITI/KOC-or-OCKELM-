# KOC+ or OCKELM+ or LSSVM+
One-class KRR or One-class KELM with Privileged Information

If you are using this code then kindly cite the folowing paper (**Paper is under review in 'Information Sciences', Elsevier**): 

[KOC+: Kernel Ridge Regression-based One-class Classification using Privileged Information](https://arxiv.org/abs/1904.08338)

Author: C. Gautam, A. Tiwari, M. Tanveer

**KOC+ can also be treated as the variant of Kernel Extreme learning Machine or Least Square SVM with zero bias. therefore paper can be named as follows:**

OCKELM+: Kernel Extreme Learning Machine based One-class Classification using Privileged Information 

or 

LSSVM+: Least Square SVM with zero bias based One-class Classification using Privileged Information


# For reproducing the results of Heart datasets:

--  Open KOC+_Heart_Experiments.ipynb in Python notebook and run all cells. It will save all results in .pkl files. Results on optimal   parameters along with optimal parameters values will be saved in a excel file.   

--  Be dfault these codes produce results for group attribute 'Age'. For other two group attributes (Sex and Electrocardiographic): change the value in cell number 3 and 4 as follows:

**For group attribute = Sex:**

Uncomment this line in cell 3:  
 privileged_space = privileged_space_tot.ix[:]['p1']

Uncomment this line in cell 4:
 feature_space = feature_space.drop('a2', axis=1)
 privfeat = 'Sex'

**For group attribute = Electrocardiographic:**

Uncomment this line in cell 3:  
 privileged_space = privileged_space_tot.ix[:]['p3']

Uncomment this line in cell 4:
 feature_space = feature_space.drop('a7', axis=1)
 privfeat = 'Elect'


# For any query, you can reach me at chandangautam31@gmail.com , phd1501101001@iiti.ac.in
