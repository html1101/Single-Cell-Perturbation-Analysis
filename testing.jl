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
@printf "File %s:\n\tRows: %d\n\tCols: %d\n\tFirst Col Names: %s\n" path nrow(dframe) ncol(dframe) first(names(dframe), 20)

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

# Now we're going to make the matrix whose rows are composed of the col values for each gene
# Ex: drug A [gene a, gene b, ..., gene n]
data = Matrix(dframe[:, Not(["Column1", "sm_name", "cell_type"])])

# (1) Perform SVD
U, sigma, V = svd(transpose(data))

# Create a rank n SVD decomposition
n = 100
rank_n_U = U[:,1:n]
rank_n_sigma = Diagonal(sigma[1:n])
rank_n_V = V[1:n,:]

# Apply model to set
Y = rank_n_V
# Rank n estimation of data
X = rank_n_U * rank_n_sigma * rank_n_V

# Display the difference between the actual data + rank-n data projection
new_comp = data - transpose(X)
@printf "Difference between data:\n\tMean: %f\n\tSTD: %f\n\tAccuracy: %f%s\n" mean(new_comp) std(new_comp) (100 - 100 * mean((data - transpose(X)) / data)) "%"

# Standardize down data
m = mean(Y)
s = std(Y)

Z = (Y .- m)

# Calculate covariance matrix
c = cov(Z)
display(heatmap(1:size(c,1),
    1:size(c,2), c,
    xlabel="Drugs", ylabel="Drugs",
    title="Drug-Gene Reaction Similarities"))

# Let's label points by their corresponding EPC (pharmacologic class)
# if available, else do not show
# (1) Load drug information
info = JSON.parsefile("drug_information_indexed.json")

# (3) Map these to colors
map = Dict(
    "Azole Antifungal [EPC]"=>"red",
    nothing=>"grey",
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

map_2 = Dict(
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
    "Corticosteroid [EPC]"=>[]
)

# (2) Get SM names + replace with 4th column
let coloring = []
let i = 0
for drug in dframe[!, "sm_name"]
    if info[drug][3] != nothing
        push!(map_2[info[drug][3]], Y[:,i+1])
        push!(coloring, map[info[drug][3]])
    end
    i += 1
end
end

# Plot first three principal components

p = scatter(Y[1,:], Y[2,:], Y[3,:], marker=:circle, markersize=2, c=coloring, linewidth=0)
plot(p,xlabel="PC1", ylabel="PC2", zlabel="PC3")
end

# (2) Get SM names + replace with 4th column
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
    end
    ii += 1
end
end

# Cluster points together
display(epc_clustering)

# Now look at average points + distance from each other
# (looking for a small STD + mean HOPEFULLY)
for (epc_class, values) in epc_clustering
    @printf("DRUG CLASS %s:\n", epc_class)
    display(mean(vec(values)))
    @printf("For drug class %s:\n\tMean: %f\n\tSTD: %f\n", epc_class, mean(mean(vec(values))), mean(std(vec(values))))
end

@printf("ALL DRUG CLASSES:\n")
display(Y)

@printf "\tMean: %f\n\tSTD: %f\n" mean(mean(vec(X))) mean(std(vec(X)))
