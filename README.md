# Oncolytic-Viral-Therapy-Efficacy-and-Biomarker-Prediction
Predicting if a patient will respond to PVSRIPO viral immunotherapy through RNA cell sequence analysis, flow cytometry and predictive biomarkers

Used feed forward neural network which outperformed convolutional neural network and recurrent neural network to predict is a patient will respond or be resistant to PVSRIPO (Oncolytic Poliovirus) immunotherapy. The data is RNA sequence data indicating tens of thousands of gene expressions. The data was curated in a glioblastoma cell line before and after immunotherapy. Model predicts with up to 100% without overfitting. 

Furthermore, an unsupervised neural network a little different than the other one was implemented and accurately predicted an important biomarker, NRP-2, which indicates resistance to the PVSRIPO therapy due to it's signaling nature for macrophages and dendritic cells in the tumor microenvironment.

Will soon be implementig feature to indicate macrophage polarization and quantification using flow cytometry t better predict a patient outcome.
