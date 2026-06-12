import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
numpy2ri.activate()

WGCNA = importr("WGCNA")
dynamicTreeCut = importr("dynamicTreeCut")
flashClust = importr("flashClust")
base = importr("base")


def build_and_evaluate_hypergraph(csv_path, out_dir, topk_roi=25, min_edge=1, max_edge=50, 
                                var_q=0.10, random_seed=42, min_module_size=10, 
                                merge_cut_height=0.25, deep_split=2, 
                                network_type="signed", cor_type="bicor", 
                                power_candidates=list(range(4, 15)), 
                                evaluate=True, traits_df=None):
    """
    Integrated hypergraph construction and evaluation function
    """
    # 初始化
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(random_seed)

    print("\n1. Preparing data...")
    df = pd.read_csv(csv_path, header=None)
    actual_roi_count = df.shape[0] 
    gene_count = df.shape[1]        
    roi_names = [f"ROI_{i}" for i in range(actual_roi_count)]  
    gene_names = [f"Gene_{i}" for i in range(gene_count)]      
    
    X = df.values.astype(np.float64)
    X = np.nan_to_num(X)
    
    # Gene filtering (keep genes with high expression variance)
    gene_vars = np.var(X, axis=0)  # compute variance per column (gene)
    var_threshold = np.quantile(gene_vars, var_q)
    keep_genes = gene_vars >= var_threshold
    X_filtered = X[:, keep_genes]  # After filtering: (samples, genes)
    filtered_gene_count = sum(keep_genes)
    print(f"Original data: {X.shape}, After filtering: {X_filtered.shape} (kept genes: {filtered_gene_count})\n")

    # Standardization (do not transpose, keep rows=samples, cols=genes)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered) 
    print(f"Scaled data shape: {X_scaled.shape} (rows=samples, cols=genes)\n")

    r_data = ro.r.matrix(
        X_scaled,
        nrow=X_scaled.shape[0], 
        ncol=X_scaled.shape[1],  
        dimnames=ro.r.list(
            ro.StrVector(roi_names),        # row names: sample names (match number of rows)
            ro.StrVector([f"Gene_{i}" for i in range(filtered_gene_count)])  # col names: gene names (match number of cols)
        )
    )

    print("Dimensions of R matrix r_data (rows, cols):", ro.r.dim(r_data))  # e.g. (116, 1219)
    ro.r.assign("datExpr", r_data)
    # ------------------------- 2. WGCNA-------------------------
    try:
        ro.r('''
        library(doParallel)
        registerDoParallel(cores=detectCores()-1)
        options(cores=detectCores()-1)
        ''')
        print(f"Parallel computation enabled (using {ro.r('detectCores()')[0]-1} cores)")
    except:
        print(f"Parallel initialization failed: {str(e)} (continuing single-threaded)")
    
    ro.r.assign("powerVector", ro.IntVector(power_candidates))
    ro.r.assign("networkType", network_type)
    
    cor_func = "WGCNA::bicor" if cor_type == "bicor" else "stats::cor"
    
    ro.r(f'''
    sft <- WGCNA::pickSoftThreshold(
        data = datExpr,
        powerVector = powerVector,
        networkType = networkType,
        corFnc = {cor_func},
        corOptions = list(use = 'p'),
        verbose = 5)
    ''')

    # Extract best soft-threshold
    ro.r('fit_indices_df <- as.data.frame(sft$fitIndices)')
    fit_indices = pd.DataFrame(
        np.array(ro.r('as.matrix(fit_indices_df)')),
        columns=list(ro.r('colnames(fit_indices_df)'))
    )
    fit_indices['Power'] = fit_indices['Power'].astype(int)
    best_power = int(fit_indices.loc[fit_indices['SFT.R.sq'].idxmax(), 'Power'])
    best_r_sq = fit_indices.loc[fit_indices['Power'] == best_power, 'SFT.R.sq'].values[0]
    ro.r.assign("best_power", best_power)
    print(f"Selected soft-threshold: {best_power} (SFT.R.sq={best_r_sq:.3f})")
    
    # Module identification
    print("Identifying modules...")
    ro.r(f'''
    adjacency <- WGCNA::adjacency(
        datExpr = datExpr,
        power = best_power,
        type = networkType,
        corFnc = {cor_func},
        corOptions = list(use = 'p'))

    tom <- WGCNA::TOMsimilarity(adjacency, TOMType=networkType, verbose=0)
    dissTOM <- 1 - tom
    
    geneTree <- flashClust::hclust(as.dist(dissTOM), method="average")
    
    dynamicMods <- dynamicTreeCut::cutreeDynamic(
        dendro = geneTree,
        distM = dissTOM,
        deepSplit = {deep_split},
        minClusterSize = {min_module_size},
        verbose = 0)
    
    moduleColors <- WGCNA::labels2colors(dynamicMods)
    
    MEs <- WGCNA::moduleEigengenes(expr = datExpr, colors = moduleColors)$eigengenes
    
    merged <- WGCNA::mergeCloseModules(
        expr = datExpr,
        MEs = MEs,
        colors = moduleColors,
        cutHeight = {merge_cut_height},
        verbose = 0)
    ''')

    print("\nViewing adjacency matrix info:")
    # Print dimensions of adjacency matrix in R
    adj_dim = ro.r("dim(adjacency)")
    print(f"Adjacency matrix dimensions (genes x genes): {adj_dim}")

    # Get module color labels
    color_labels = np.array(ro.r("as.character(merged$colors)"), dtype=str)
    unique_colors = [c for c in np.unique(color_labels) if c != "grey"]
    print(f"Detected {len(unique_colors)} modules (excluding grey)")

    # ------------------------- 3. Build hyperedges -------------------------
    print("\n3. Building hyperedges...")
    hyperedges = []
    module_info = {}
    
    for color in unique_colors:

        mod_genes = np.where(color_labels == color)[0]
        if len(mod_genes) < min_module_size:
            continue
            

        mod_expr = X_scaled[:, mod_genes].mean(axis=1)
        top_roi_indices = np.argsort(mod_expr)[-topk_roi:] 
        
        if min_edge <= len(top_roi_indices) <= max_edge:
            hyperedges.append(top_roi_indices)
            module_info[color] = {
                "size": len(mod_genes),
                "top_roi": [roi_names[i] for i in top_roi_indices],
                "gene_count": len(mod_genes)
            }
            print(f"{color} module: genes={len(mod_genes)}, hyperedge size={len(top_roi_indices)}")

    if len(hyperedges) == 0:
        print("Warning: No hyperedges constructed, skipping saving")
        return None, None, None
    
    print("\n4. Saving results...")
    H_np = np.zeros((actual_roi_count, len(hyperedges)), dtype=int)
    for e, roi_indices in enumerate(hyperedges):
        H_np[roi_indices, e] = 1
    
    hypergraph_df = pd.DataFrame(
        H_np,
        index=roi_names,
        columns=[f"hyperedge_{i}" for i in range(len(hyperedges))]
    )
    hypergraph_df.index.name = "ROI_ID"
    hypergraph_path = os.path.join(out_dir, "hypergraph_matrix.csv")
    hypergraph_df.to_csv(hypergraph_path)
    print(f"Hypergraph matrix saved to {hypergraph_path}")


    weights_df = pd.DataFrame(
        np.ones(len(hyperedges)),
        index=[f"hyperedge_{i}" for i in range(len(hyperedges))],
        columns=["weight"]
    )
    weights_df.index.name = "hyperedge"
    weights_path = os.path.join(out_dir, "hyperedge_weights.csv")
    weights_df.to_csv(weights_path)
    print(f"Hyperedge weights saved to {weights_path}")


    module_info_path = os.path.join(out_dir, "module_info.json")
    with open(module_info_path, "w") as f:
        json.dump(module_info, f, indent=2)
    print(f"Module info saved to {module_info_path}")


    params = {
        "CSV_PATH": csv_path,
        "TOPK_ROI": topk_roi,
        "BEST_POWER": best_power,
        "CORRELATION_TYPE": cor_type,
        "ACTUAL_ROI_COUNT": actual_roi_count,
        "WGCNA_PARAMS": {
            "MIN_MODULE_SIZE": min_module_size,
            "MERGE_CUT_HEIGHT": merge_cut_height,
            "DEEP_SPLIT": deep_split,
            "NETWORK_TYPE": network_type
        }
    }
    params_path = os.path.join(out_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved to {params_path}")


    if evaluate and len(unique_colors) > 0:
        print("\n5. Module quality evaluation...")
        try:
            # 检查是否有有效模块
            if len(unique_colors) == 0 or (len(unique_colors) == 1 and "grey" in unique_colors):
                print("No valid modules for evaluation, skipping evaluation step")
                return hypergraph_df, weights_df, module_info
            
            ro.r('''
            MEs_merged <- WGCNA::moduleEigengenes(expr = datExpr, colors = merged$colors)$eigengenes
            ''')
            ro.r('''
            kME_all <- WGCNA::signedKME(datExpr, MEs_merged)
            ''')         
            ro.r('''
            imc <- WGCNA::intramodularConnectivity(adjacency, merged$colors)
            rownames(imc) <- rownames(adjacency)  
            ''')
            

            kME_df = pd.DataFrame(
                np.array(ro.r('as.matrix(kME_all)')),
                columns=list(ro.r('colnames(kME_all)')),
                index=list(ro.r('rownames(kME_all)'))
            )
            
            imc_df = pd.DataFrame(
                np.array(ro.r('as.matrix(imc)')),
                columns=['kTotal', 'kWithin', 'kOut', 'kDiff'],
                index=kME_df.index  
            )

            gene_colors = pd.Series(color_labels, index=kME_df.index, name='module_color')
            

            own_kME = []
            for gene, color in gene_colors.items():
                col_name = f'kME{color}'
                if col_name in kME_df.columns:
                    own_kME.append(kME_df.at[gene, col_name])
                else:
                    own_kME.append(np.nan)
            

            gene_metrics = pd.DataFrame({
                'module_color': gene_colors,
                'kME_in_own_module': pd.Series(own_kME, index=kME_df.index),
                'kWithin': imc_df['kWithin']
            })
    
            module_summary = gene_metrics.groupby('module_color').agg({
                'kME_in_own_module': ['mean', 'median'],
                'kWithin': 'mean',
                'module_color': 'size'
            })
            module_summary.columns = ['kME_mean', 'kME_median', 'kWithin_mean', 'module_size']
            module_summary = module_summary.sort_values('module_size', ascending=False)
            
            gene_metrics_path = os.path.join(out_dir, "gene_level_metrics.csv")
            gene_metrics.to_csv(gene_metrics_path)
            print(f"Gene-level metrics saved to {gene_metrics_path}")
            
            module_summary_path = os.path.join(out_dir, "module_summary.csv")
            module_summary.to_csv(module_summary_path)
            print(f"Module-level summary saved to {module_summary_path}")
            print("Module quality evaluation completed")
        except Exception as e:
            print(f"Module evaluation failed: {str(e)}")
    
    return hypergraph_df, weights_df, module_info

if __name__ == "__main__":

    '''
    "topk_roi":                 
    "min_edge":                 
    "max_edge":                 
    "var_q":                    
    "min_module_size":          
    "merge_cut_height":         
    "deep_split":               
    "power_candidates":         
    '''

    params = {
        "csv_path": "../Transcriptome-driven_hypergraph_construction/AHBA_AD.csv",
        "out_dir": "../Transcriptome-driven_hypergraph_construction/hypergraph_AD_wgcna",
        "topk_roi": 20,
        "min_edge": 10,
        "max_edge": 50,
        "var_q": 0.10,
        "random_seed": 42,
        "min_module_size": 10,
        "merge_cut_height": 0.25,
        "deep_split": 2,
        "network_type": "signed",
        "cor_type": "bicor",
        "power_candidates": list(range(1, 30)),
        "evaluate": True,
        "traits_df": None
    }
    
    build_and_evaluate_hypergraph(**params)
