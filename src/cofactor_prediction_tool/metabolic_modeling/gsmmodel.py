import copy

from cobra.io import read_sbml_model, write_sbml_model
import pandas as pd
import json

def get_gene_cofactors(model, predictions, cofactors_map):
    gene_cofactors = {}
    for reaction in model.reactions:
        gene_ids = [gene.id for gene in reaction.genes if gene.id in predictions.index]
        gene_predictions = predictions.loc[gene_ids]
        predicted_cofactors = [cofactor for cofactor in cofactors_map.keys() if gene_predictions[cofactor].sum() > 0]
        if predicted_cofactors:
            for cofactor, pairs in cofactors_map.items():
                for pair in pairs:
                    if set(pair).issubset({met.id for met in reaction.metabolites.keys()}):
                        for gene in gene_ids:
                            if gene not in gene_cofactors:
                                gene_cofactors[gene] = set()
                            gene_cofactors[gene].add(cofactor)
    return gene_cofactors

def get_cofactor_for_reaction(reaction, cofactors_map):
    cofactors = []
    for cofactor, pairs in cofactors_map.items():
        for pair in pairs:
            if set(pair).issubset({met.id for met in reaction.metabolites.keys()}):
                cofactors.append(pair)
    if len(cofactors)>1:
        print(f"Multiple cofactors found for reaction {reaction.id}")
    if len(cofactors)==0:
        return None
    return cofactors[0]

def update_model(config_file, **kwargs):
    report = {'match':0,'mismatch':0, 'match_percent':0, 'mismatch_percent':0}    
    params = json.load(open(config_file))
    model_path = kwargs.get("model_path", params["model_path"])
    predictions = pd.read_csv(kwargs.get("predictions", params["predictions"]), index_col=0, sep="\t")
    cofactors_map = params["cofactors_map"]
    drop_non_predicted_reactions = params["drop_non_predicted_reactions"]
    model = read_sbml_model(model_path)
    metabolites = {met.id for met in model.metabolites}
    cofactors_map_filter = {}
    #cofactors_map = {key: value for key, value in cofactors_map.items() if set([pair[0] for pair in value]).issubset(metabolites) and set([pair[1] for pair in value]).issubset(metabolites)}
    for key, value in cofactors_map.items():
        for pair in value:
            if pair[0] in metabolites and pair[1] in metabolites:
                if key not in cofactors_map_filter:
                    cofactors_map_filter[key] = []
                cofactors_map_filter[key].append(pair)
    gene_cofactors = get_gene_cofactors(model, predictions, cofactors_map_filter)
    added_reactions = set()
    to_remove = set()
    for gene in model.genes:
        if gene.id in predictions.index and gene.id in gene_cofactors:
            cofactors_predicted = [cofactor for cofactor in cofactors_map_filter.keys() if predictions.loc[gene.id, cofactor] > 0]
            difference = set(cofactors_predicted) - gene_cofactors[gene.id]
            report["match"] += len(set(cofactors_predicted) & gene_cofactors[gene.id])
            report["mismatch"] += len(set(cofactors_predicted) - gene_cofactors[gene.id])
            if difference:
                for reaction in set(gene.reactions) - added_reactions:
                    cofactors = get_cofactor_for_reaction(reaction, cofactors_map_filter)
                    if cofactors:
                        for tmp in difference:
                            new_reaction = copy.deepcopy(reaction)
                            if drop_non_predicted_reactions:
                                to_remove.add(reaction.id)
                            new_reaction_id = f"{reaction.id}_{tmp}"
                            if not {new_reaction_id}.issubset(set([reaction.id for reaction in model.reactions])):
                                new_reaction.id = new_reaction_id
                                model.add_reactions([new_reaction])
                                added_reactions.add(new_reaction)
                                compartments = []
                                cofactor_st = []
                                for cofactor in cofactors:
                                    st = reaction.metabolites[model.metabolites.get_by_id(cofactor)]
                                    cofactor_st.append(st)
                                    compartments.append(model.metabolites.get_by_id(cofactor).compartment)
                                    new_reaction.add_metabolites({cofactor: -st})
                                cofactors_to_add = [pair for pair in cofactors_map_filter[tmp] if model.metabolites.get_by_id(pair[0]).compartment in compartments]
                                for pair in cofactors_to_add:
                                    new_reaction.add_metabolites({pair[0]: cofactor_st[0], pair[1]: cofactor_st[1]})

    report["match_percent"] = report["match"]/(report["match"]+report["mismatch"])
    report["mismatch_percent"] = report["mismatch"]/(report["match"]+report["mismatch"])
    
    with open("report.json", "w") as f:
        json.dump(report, f)

    model.remove_reactions(list(to_remove))
    write_sbml_model(model, f"updated_model.xml")




if __name__ == '__main__':
    update_model ("/home/jgoncalves/cofactor_prediction_tool/nextflow/config.json")
