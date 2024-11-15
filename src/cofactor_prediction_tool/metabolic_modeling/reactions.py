import os
import pandas as pd
from cobra.io import read_sbml_model

def get_cofactor_for_reaction(reaction, cofactors_map):
    """
    Get the cofactors for a given reaction.

    Parameters:
    reaction (Reaction): The reaction for which to find the cofactors.
    cofactors_map (dict): A dictionary mapping cofactors to pairs of metabolites.

    Returns:
    list: The list of cofactors found for the reaction, otherwise an empty list.
    """
    cofactors = []
    for cofactor, pairs in cofactors_map.items():
        for pair in pairs:
            if set(pair).issubset({met.id for met in reaction.metabolites.keys()}):
                cofactors.append(cofactor)
    return cofactors

def count_reactions_with_cofactors(model, cofactors_map):
    """
    Count the total number of reactions and reactions with cofactors in the model.

    Args:
        model (Model): The metabolic model.
        cofactors_map (dict): A dictionary mapping cofactors to their corresponding pairs.

    Returns:
        tuple: Total reactions and the number of reactions with cofactors.
    """
    total_reactions = len(model.reactions)
    reactions_with_cofactors = 0

    for reaction in model.reactions:
        cofactors = get_cofactor_for_reaction(reaction, cofactors_map)
        if cofactors:
            reactions_with_cofactors += 1

    return total_reactions, reactions_with_cofactors

def analyze_model(model_path, cofactors_map):
    """
    Load the model, count reactions with cofactors, and return results for a summary report.

    Args:
        model_path (str): Path to the SBML model file.
        cofactors_map (dict): Dictionary of cofactors and their respective metabolite pairs.

    Returns:
        dict: Summary of the model, including model name, total reactions, and reactions with cofactors.
    """
    # Load the SBML model
    model = read_sbml_model(model_path)

    # Count total reactions and reactions involving cofactors
    total_reactions, reactions_with_cofactors = count_reactions_with_cofactors(model, cofactors_map)

    # Extract the model name from the file path
    model_name = os.path.basename(model_path).split('.')[0]

    return {
        "Model Name": model_name,
        "Total Number of Reactions": total_reactions,
        "Number of Reactions with Cofactors": reactions_with_cofactors
    }

def analyze_all_models_in_directory(directory_path, cofactors_map, output_csv):
    """
    Analyze all SBML models (XML files) in a given directory and save a summary to a CSV.

    Args:
        directory_path (str): Path to the directory containing SBML models.
        cofactors_map (dict): Dictionary of cofactors and their respective metabolite pairs.
        output_csv (str): Path to the CSV file to save results.

    Returns:
        None
    """
    results = []
    
    # Analyze each XML model in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.xml'):
            model_path = os.path.join(directory_path, file_name)
            result = analyze_model(model_path, cofactors_map)
            results.append(result)

    # Create a DataFrame and save the results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Summary CSV saved to: {output_csv}")

if __name__ == '__main__':
    # Path to the directory containing SBML models
    models_directory = "/home/jgoncalves/cofactor_prediction_tool/case_studies/inputs/"
    
    # Output CSV path
    output_csv = "/home/jgoncalves/cofactor_prediction_tool/case_studies/cofactor_summary.csv"
    
    # Define cofactors map
    cofactors_map = {
        "NAD": [["nad_c", "nadh_c"], ["nad_m", "nadh_m"], ["nad_p", "nadh_p"], ["nad_r", "nadh_r"], ["nad_x", "nadh_x"], ["nad_l", "nadh_l"], ["nad_n", "nadh_n"],
                ["nad_e", "nadh_e"], ["nad_g", "nadh_g"], ["nad_h", "nadph_h"]],
        "NADP": [["nadp_c", "nadph_c"], ["nadp_m", "nadph_m"], ["nadp_p", "nadph_p"], ["nadp_r", "nadph_r"], ["nadp_x", "nadph_x"], ["nadp_l", "nadph_l"],
                 ["nadp_n", "nadph_n"], ["nadp_e", "nadph_e"], ["nadp_g", "nadph_g"], ["nadp_h", "nadph_h"]],
        "FAD": [["fad_c", "fadh2_c"], ["fad_m", "fadh2_m"], ["fad_p", "fadh2_p"], ["fad_r", "fadh2_r"], ["fad_x", "fadh2_x"], ["fad_l", "fadh2_l"],
                ["fad_n", "fadh2_n"], ["fad_e", "fadh2_e"], ["fad_g", "fadh2_g"], ["fad_h", "fadh2_h"]],
        "SAM": [["amet_c", "ahcys_c"], ["amet_m", "ahcys_m"], ["amet_p", "ahcys_p"], ["amet_r", "ahcys_r"], ["amet_x", "ahcys_x"], ["amet_l", "ahcys_l"],
                ["amet_n", "ahcys_n"], ["amet_e", "ahcys_e"], ["amet_g", "ahcys_g"], ["amet_h", "ahcys_h"]],
        "CoA": [["coa_c", "accoa_c"], ["coa_c", "succoa_c"], ["coa_c", "ppcoa_c"], ["coa_m", "accoa_m"], ["coa_m", "succoa_m"], ["coa_m", "ppcoa_m"],
                ["coa_p", "accoa_p"], ["coa_p", "succoa_p"], ["coa_p", "ppcoa_p"], ["coa_r", "accoa_r"], ["coa_r", "succoa_r"], ["coa_r", "ppcoa_r"],
                ["coa_x", "accoa_x"], ["coa_x", "succoa_x"], ["coa_x", "ppcoa_x"], ["coa_l", "accoa_l"], ["coa_l", "succoa_l"], ["coa_l", "ppcoa_l"],
                ["coa_n", "accoa_n"], ["coa_n", "succoa_n"], ["coa_n", "ppcoa_n"], ["coa_e", "accoa_e"], ["coa_e", "succoa_e"], ["coa_e", "ppcoa_e"],
                ["coa_g", "accoa_g"], ["coa_g", "succoa_g"], ["coa_g", "ppcoa_g"], ["coa_h", "accoa_h"], ["coa_h", "succoa_h"], ["coa_h", "ppcoa_h"]],
        "THF": [["thf_c", "mlthf_c"], ["dhf_c", "mlthf_c"], ["thf_m", "mlthf_m"], ["dhf_m", "mlthf_m"], ["thf_p", "mlthf_p"], ["dhf_p", "mlthf_p"],
                ["thf_r", "mlthf_r"], ["dhf_r", "mlthf_r"], ["thf_x", "mlthf_x"], ["dhf_x", "mlthf_x"], ["thf_l", "mlthf_l"], ["dhf_l", "mlthf_l"],
                ["thf_n", "mlthf_n"], ["dhf_n", "mlthf_n"], ["thf_e", "mlthf_e"], ["dhf_e", "mlthf_e"], ["thf_g", "mlthf_g"], ["dhf_g", "mlthf_g"],
                ["thf_h", "mlthf_h"], ["dhf_h", "mlthf_h"]],
        "FMN": [["fmn_c", "fmnh2_c"], ["fmn_m", "fmnh2_m"], ["fmn_p", "fmnh2_p"], ["fmn_r", "fmnh2_r"], ["fmn_x", "fmnh2_x"], ["fmn_l", "fmnh2_l"],
                ["fmn_n", "fmnh2_n"], ["fmn_e", "fmnh2_e"], ["fmn_g", "fmnh2_g"], ["fmn_h", "fmnh2_h"]],
        "GSH": [["gthrd_c", "gthox_c"], ["gthrd_m", "gthox_m"], ["gthrd_p", "gthox_p"], ["gthrd_r", "gthox_r"], ["gthrd_x", "gthox_x"],
                ["gthrd_l", "gthox_l"], ["gthrd_n", "gthox_n"], ["gthrd_e", "gthox_e"], ["gthrd_g", "gthox_g"], ["gthrd_h", "gthox_h"]],
        "Ferredoxin": [["fdxox_c", "fdxrd_c"], ["fdxox_m", "fdxrd_m"], ["fdxox_p", "fdxrd_p"], ["fdxox_r", "fdxrd_r"], ["fdxox_x", "fdxrd_x"],
                       ["fdxox_l", "fdxrd_l"], ["fdxox_n", "fdxrd_n"], ["fdxox_e", "fdxrd_e"], ["fdxox_g", "fdxrd_g"], ["fdxox_h", "fdxrd_h"]],
        "Ferricytochrome": [["cytcox_c", "cytcrd_c"], ["cytcox_m", "cytcrd_m"], ["cytcox_p", "cytcrd_p"], ["cytcox_r", "cytcrd_r"],
                            ["cytcox_x", "cytcrd_x"], ["cytcox_l", "cytcrd_l"], ["cytcox_n", "cytcrd_n"], ["cytcox_e", "cytcrd_e"], ["cytcox_g", "cytcrd_g"],
                            ["cytcox_h", "cytcrd_h"]],
        "Ubiquinone": [["q8_c", "q8h2_c"], ["q8_m", "q8h2_m"], ["q8_p", "q8h2_p"], ["q8_r", "q8h2_r"], ["q8_x", "q8h2_x"], ["q8_l", "q8h2_l"],
                       ["q8_n", "q8h2_n"], ["q8_e", "q8h2_e"], ["q8_g", "q8h2_g"], ["q8_h", "q8h2_h"]],
        "Menaquinone": [["mqn8_c", "mqn8h2_c"], ["mqn8_m", "mqn8h2_m"], ["mqn8_p", "mqn8h2_p"], ["mqn8_r", "mqn8h2_r"], ["mqn8_x", "mqn8h2_x"],
                        ["mqn8_l", "mqn8h2_l"], ["mqn8_n", "mqn8h2_n"], ["mqn8_e", "mqn8h2_e"], ["mqn8_g", "mqn8h2_g"], ["mqn8_h", "mqn8h2_h"]],
        "Plastoquinone": [["pq6_c", "pq6h2_c"], ["pq6_m", "pq6h2_m"], ["pq6_p", "pq6h2_p"], ["pq6_r", "pq6h2_r"], ["pq6_x", "pq6h2_x"],
                          ["pq6_l", "pq6h2_l"], ["pq6_n", "pq6h2_n"], ["pq6_e", "pq6h2_e"], ["pq6_g", "pq6h2_g"], ["pq6_h", "pq6h2_h"]]
    }

    # Analyze all models in the specified directory and save results to a CSV
    analyze_all_models_in_directory(models_directory, cofactors_map, output_csv)
