$$\documentclass{article}

\usepackage{float}
\usepackage{graphicx} % Required for inserting images
\usepackage{minted}
\usepackage{biblatex}
\usepackage{amsmath}
\usepackage{epsfig}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{amstext}
\usepackage{xspace}
\usepackage{theorem}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{xspace}
\addbibresource{citations (2).bib}

% needed for nomal spacing
\makeatletter
 \setlength{\textwidth}{6in}
 \setlength{\oddsidemargin}{0in}
 \setlength{\evensidemargin}{0.5in}
 \setlength{\topmargin}{0in}
 \setlength{\textheight}{9in}
 \setlength{\headheight}{0pt}
 \setlength{\headsep}{0pt}
 \setlength{\marginparwidth}{59pt}

 \setlength{\parindent}{0pt}
 \setlength{\parskip}{5pt plus 1pt}
 \setlength{\theorempreskipamount}{5pt plus 1pt}
 \setlength{\theorempostskipamount}{0pt}
 \setlength{\abovedisplayskip}{8pt plus 3pt minus 6pt}

\title{21-241 Final Project: Analyzing Drug Response Using PCA}
\author{Sarah Cross, Aashna Kulshrestha}
\date{December 8th, 2023}

\begin{document}
\newcounter{mynum}
\newcommand\mycite[2]{[#1]\setcounter{mynum}{0}\refstepcounter{mynum}\label{#2}}
\newcommand\mybib[1]{\item[\ref{#1}]\bibentry{#1}}

\maketitle

\section{Overview}
Analyzing and predicting how small molecules change gene expression across different cell types is a remarkably complex problem that is extremely important in the field of drug discovery, as new medicines are typically costly and extremely expensive to develop. If able to accurately predict how a small molecule upregulates or downregulates a set of genes, therapeutics could be developed to target genes known to be associated with specific illnesses entirely computationally, significantly accelerating and expending the development of new medicines.

This project, using classic Linear Algebra techniques, attempts to divulge certain insights into the mechanisms and patterns found in drug perturbation, looking in particular at how to reduce down the dimensionality of differential gene expression using PCA in order to observe patterns amongst similar drugs. Additionally, by obtaining the principal components of a set of 18,211 genes, this may serve as a better input for a model predicting drug perturbation, as models being trained on relatively few samples (634 samples, in this case) with many inputs (18,211 genes) are likely to be subjected to extreme overfitting (eg. the models have enough parameters that they can "remember" the exact points given during training, and retain high accuracy during training while receiving extremely low accuracy when exposed to new data).

\section{Background}
\subsection{Biological Background}
This project uses a dataset developed by \textit{Open Problems in Single-Cell Analysis}, containing single-cell gene expression profiles (with 18,211 genes) taken from human peripheral blood mononuclear cells (PBMCs). The cells were treated with 144 compounds, then gene expression was measured and post-processed \mycite{6}{Open Problems}.

The researchers stipulated that PCA was likely to perform well on this particular problem, as many genes are co-regulated. This can happen for a variety of reasons; for example, genes may be grouped into the same operon and may be transcribed into proteins together, or may be part of the same metabolic pathway and coregulate one another \mycite{2}{Piyush} \mycite{9}{Piyush}. As a result, it is likely that there are not 18,211 genes that are changing completely independently of one another, and that many are linearly dependent on one another, making principal component analysis a viable pre-processing technique for this dataset in particular \mycite{4}{Gene regulation} \mycite{3}{Encyclopedia}.

\subsection{Linear Algebra Background}
The first step to performing PCA (Principal Component Analysis) is to pre-process and center the relevant data from a chosen dataset. This ensures that, rather than capturing the overall average as a significant variable, the resulting components are only looking at variance from \textit{within} the dataset. In terms of this project, the dataset chosen was already centered.

Because different cell types inherently respond differently to small molecules, the researchers filtered the data based on cell type, focusing specifically on the NK and T-regulatory cell types individually, as the most data was available for these specific types of cells (not all drugs were tested on all cells because of the cytotoxic effects of certain drugs on certain cells). Both the NK and T-regulatory cells contained 146 available drug samples, with 18211 genes tested. As such, we filtered the dataset by these cell types, leaving us with A, a \textit{m x n} matrix where \textit{m = 18211} and \textit{n = 146} for each cell type.

Next, we must find the Singular Value Decomposition (SVD) of A.
 \[A = \begin{pmatrix} x_1 \\ \dots \\ x_n \end{pmatrix}\]

  \[U \Sigma V^T= \begin{pmatrix} & & \\ u_1 &\dots & u_m \\ & & \end{pmatrix} \begin{pmatrix} \sigma_1 & & \\ & \ddots & \\ & & \sigma_n\end{pmatrix} \begin{pmatrix} v_1^T \\ \dots \\ v_n^T \end{pmatrix} \]

To continue performing PCA, we must choose a rank k approximation. Our thought process for choosing an appropriate k was based on the reconstruction accuracy for the dataset. Through trial-and-error, we found that when \textit{k = 100}, the original dataset can be reconstructed with a 99.4\% accuracy. Due to the high accuracy rate, we determined our \textit{k} to be 100.

SVD according to the rank \textit{k} approximation: \[U \Sigma V^T= \begin{pmatrix} & & \\ u_1 &\dots & u_k \\ & & \end{pmatrix} \begin{pmatrix} \sigma_1 & & \\ & \ddots & \\ & & \sigma_k\end{pmatrix} \begin{pmatrix} v_1^T \\ \dots \\ v_k^T \end{pmatrix} \]

Finally, we can plot the first three principal components to obtain a summarized version of the dataset. The results of our findings are shown in \textit{Section 4: Graphs and Results}.

\section{Code}
\textit{Note: the @printf command does not allow for new lines; however, for readability, we placed each argument on a new line.}
\begin{minted}{julia}
# Import relevant packages
import Pkg;
Pkg.add("DataFrames")
Pkg.add("Statistics")
Pkg.add("LinearAlgebra")
Pkg.add("CSV")
Pkg.add("JSON")
using DataFrames, Statistics, Plots
using LinearAlgebra, Printf, CSV, DelimitedFiles, JSON

# Read file + push into dataframe
path = "nk_cells.csv"

dframe = DataFrame(CSV.File(path, normalizenames=true))

# Display information about the file we are looking at
@printf "File %s:\n\tRows: %d\n\tCols: %d\n\tFirst Col Names: %s"
    path
    nrow(dframe)
    ncol(dframe)
    first(names(dframe), 20)

# There are several types of cells in the files:
"""
- 275 B cells - produce antibody molecules - lymph
- 97  T cells CD4+ - type of T cell, releases cytokines to activate immune cells
("helper cells") - myeloid
- 446 T regulatory cells - type of T cell, modulate immune system +
prevent autoimmune disease by maintaining tolerance to self-antigens
(essentially prevent body from attacking itself) - myeloid
- 405 NK cells - part of innate immune system, "natural killer cells",
allows body to fight back wo any previous "memory" of a pathogen - lymph
- 172 T cells CD8+ - type of T cell, cytoxic T cell ("killer cells")  - myeloid
"""
# We've selected NK cells in particular.

# Now we're going to make the matrix whose rows are composed of the col values for
# each gene
# Ex: drug A [gene a, gene b, ..., gene n]
data = Matrix(dframe[:, Not(["Column1", "sm_name", "cell_type"])])

# (1) Perform SVD
U, sigma, V = svd(transpose(data))

# Create a rank n SVD decomposition
k = 100
rank_k_U = U[:,1:k]
rank_k_sigma = Diagonal(sigma[1:k])
rank_k_V = V[1:k,:]

# Apply model to set
Y = rank_k_V
# Rank k estimation of data
X = rank_k_U * rank_k_sigma * rank_k_V

# Display the difference between the actual data + rank-n data projection
new_comp = data - transpose(X)
@printf "Difference between data:\n\tMean: %f\n\tSTD: %f\n\tAccuracy: %f%%"
    mean(new_comp)
    std(new_comp)
    (100 - 100 * mean((data - transpose(X)) / data))

# Standardize down data
m = mean(Y)
s = std(Y)

Z = (Y .- m)

# Calculate covariance matrix
c = cov(Z)
heatmap(1:size(c,1),
    1:size(c,2), c,
    xlabel="Drugs", ylabel="Drugs",
    title="Drug-Gene Reaction Similarities")

c = c - Matrix(1I, size(c,1), size(c,1))
display(max)
display(argmax(c))
display(dframe[!, "sm_name"][argmax[c](1)])
display(dframe[!, "sm_name"][argmax[c](2)])

# Let's label points by their corresponding EPC (pharmacologic class)
# if available, else do not show
# (1) Load drug information
info = JSON.parsefile("drug_information_indexed.json")

# (3) Map these to colors
map = Dict(
    "Azole Antifungal [EPC]"=>"red",
    nothing=>colorant"transparent",
    "Kinase Inhibitor [EPC]"=>"green",
    "Hepatitis B Virus Nucleoside Analog Reverse Transcriptase Inhibitor [EPC]"=>"purple",
    "Androgen Receptor Inhibitor [EPC]"=>"thistle",
    "Histone Deacetylase Inhibitor [EPC]"=>"indigo",
    "Dipeptidyl Peptidase 4 Inhibitor [EPC]"=>"dodgerblue",
    "Aldehyde Dehydrogenase Inhibitor [EPC]"=>"olive",
    "Alkaloid [EPC]"=>"cyan",
    "Antimycobacterial [EPC]"=>"blue",
    "Nucleoside Metabolic Inhibitor [EPC]"=>"pink",
    "Antimetabolite [EPC]"=>"orange",
    "Thalidomide Analog [EPC]"=>"yellow",
    "Histamine-1 Receptor Antagonist [EPC]"=>"violet",
    "Soluble Guanylate Cyclase Stimulator [EPC]"=>"aquamarine",
    "Corticosteroid [EPC]"=>"lightcoral"
)

# (2) Get SM names + replace with 4th column
coloring = []
for drug in dframe[!, "sm_name"]
    if info[drug][3] != nothing
        push!(coloring, map[info[drug][3]])
    end
end

# Plot first three principal components

p = scatter(Y[1,:], Y[2,:], Y[3,:], marker=:circle, markersize=2, c=coloring, linewidth=0)
plot(p,xlabel="PC1", ylabel="PC2", zlabel="PC3")

# (2) Get SM names + replace with 4th column
coloring = []
correlation = []
Y_cutdown = []

# "Cluster" values that are of the same EPC together
# classification class are of a similar PCA
epc_clustering = Dict(
    "Azole Antifungal [EPC]"=>[],
    nothing=>[],
    "Kinase Inhibitor [EPC]"=>[],
    "Hepatitis B Virus Nucleoside Analog Reverse Transcriptase Inhibitor [EPC]"=>[],
    "Androgen Receptor Inhibitor [EPC]"=>[],
    "Histone Deacetylase Inhibitor [EPC]"=>[],
    "Dipeptidyl Peptidase 4 Inhibitor [EPC]"=>[],
    "Aldehyde Dehydrogenase Inhibitor [EPC]"=>[],
    "Alkaloid [EPC]"=>[],
    "Antimycobacterial [EPC]"=>[],
    "Nucleoside Metabolic Inhibitor [EPC]"=>[],
    "Antimetabolite [EPC]"=>[],
    "Thalidomide Analog [EPC]"=>[],
    "Histamine-1 Receptor Antagonist [EPC]"=>[],
    "Soluble Guanylate Cyclase Stimulator [EPC]"=>[],
    "Corticosteroid [EPC]"=>[],
)

let ii = 1
for drug in dframe[!, "sm_name"]
    push!(correlation, info[drug][3])
    push!(epc_clustering[info[drug][3]], X[ii,:])
    if info[drug][3] !== nothing
        push!(Y_cutdown, Y[:,ii])
        push!(coloring, map[info[drug][3]])
    end
    ii += 1
end
end

# Let's label points by their corresponding EPC (pharmacologic class)
# if available, else grey

plot(xlabel="PC1",
    ylabel="PC2",
    zlabel="PC3",
    title="Drug Sample PCA - NK Cells",
    legend = :outertopright)

for (name, Y) in map_2
    if name == nothing
        continue
    end
    # Get principal components
    a, b, c = [], [], []
    for i in Y
        push!(a, i[1])
        push!(b, i[2])
        push!(c, i[3])
    end
    scatter!(
    a, b, c,
    marker=:circle,
    markersize=2,
    markerstrokewidth=0,
    label=first(name, 15) * "...", color=map[name],
    linewidth=0)
end

display(plot!())

# Preview expression profiles, broken up by drug class.
display(epc_clustering)

# Now look at average points + distance from each other
for (epc_class, values) in epc_clustering
    @printf "DRUG CLASS %s:\n" epc_class
    display(mean(vec(values)))
    @printf "For drug class %s:\n\tMean: %f\n\tSTD: %f\n"
        epc_class
        mean(mean(vec(values)))
        mean(std(vec(values)))
end

@printf("ALL DRUG CLASSES:\n")
display(Y)
@printf("\tMean: %f\n\tSTD: %f\n", mean(mean(vec(X))), mean(std(vec(X))))
\end{minted}

\section{Graphs and Results}
\includegraphics[width=1\linewidth]{nk_cells.png}
The graphs above displays the first three principal components of drug samples with pharmacological classifications (EPC) available in the OpenFDA Drug database \mycite{1}{FDA} and cross-referenced with the National Drug Codes List \mycite{5}{National Drug}, filtered by the type of cell (as different types of cells may respond very differently to the same type of drug, the researchers opted to analyze cell types separately, focusing on NK cells in particular). Ultimately it was found that there was a relatively weak correlation between drugs of the same pharmacological class, indicating that, although the drugs perform in the same way, they do not necessarily modify similar genes (however, only around 20 drugs had a corresponding EPC, which is likely not enough data to form a strong conclusion).

For NK Cells, the following information was found (in which standard deviation measures how different, on average, the gene expression data was across the drugs tested in that particular drug class):
\begin{table}
    \centering
    \begin{tabular}{cc}
 \textbf{Drug Class}&\textbf{Standard Deviation}\\
         His-1 Receptor Antagonist&  0.475053\\
         Dipeptidyl Peptidase 4 Inhibitor&  0.738885\\
         Histone Deacetylase Inhibitor&  0.868306\\
         Antimetabolite&  0.900148\\
         Nucleoside Metabolic Inhibitor&  1.317649\\
         Alkaloid&  0.660692\\
         Azole Antifungal&  0.612870\\
 Aldehyde Dehydrogenase Inhibitor& 2.026750\\
 Thalidomide Analog&0.790018\\
 Hepatitis B Virus Nucleoside Analog Reverse Transcriptase Inhibitor&1.174471\\
 Kinase Inhibitor&1.041723\\
 Androgen Receptor Inhibitor&0.727339\\
 Antimycobacterial&0.919412\\
 Soluble Guanylate Cyclase Stimulator&0.476024\\
 \textbf{All Drugs}&1.904062\\
    \end{tabular}
    \caption{Table comparing how different drug classes' resulting gene expression profiles are.}
    \label{tab:my_label}
\end{table}
Excluding Azoles and Aldehyde drugs, the following appears to indicate that drugs within the same classification generally will result in similar gene expression profiles compared to how similar they are compared to drugs of all classes.

\includegraphics[width=1\linewidth]{drug_drug_correlation.png}
The graph above displays the covariance matrix for each drug and how similarly genes respond to perturbation. Lighter colors indicate that the drugs share similar differential gene expression data (the diagonal displays values of 1 because a drug will share 100\% similarity with itself). From the data given, the covariance matrix displayed that the Vandetanib \mycite{8}{Vandetanib} and Idelalisib \mycite{7}{Idelalisib} drugs had the most similar gene expression profiles; this corroborates with existing information about both molecules, as both are anti-cancer drugs, kinase inhibitors, and both have a two-ring carbon, as well as 7 Hydrogen bond acceptor locations.

It was additionally found that a rank \textbf{100} matrix was able to form the original matrix with 99.5\% accuracy, compared to the original 18,211 genes given, indicating that many genes can be represented as linear combinations of one another, and that the data can be significantly summarized down to enable more efficient analysis and effective ML model training.

\section{Bibliography}
\nocite{*}
\printbibliography

\end{document}$$
