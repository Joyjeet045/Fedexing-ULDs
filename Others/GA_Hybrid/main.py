from utils import *
from genetic_new import *
import pandas as pd
import re
random.seed(42)


# Function to extract data from .txt file to the required format
def extract_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    uld_data = []
    package_data = []
    
    # Define regex patterns
    uld_pattern = re.compile(r'^U\d+,\d+,\d+,\d+,\d+')
    package_pattern = re.compile(r'^P-\d+,\d+,\d+,\d+,\d+,(Priority|Economy),[-\d]+')
    
    # Flags to determine the section
    uld_section = False
    package_section = False
    
    for line in lines:
        line = line.strip()
        
        # Detect sections
        if line.startswith("ULD attributes"):
            uld_section = True
            package_section = False
            continue
        elif line.startswith("Package attributes"):
            uld_section = False
            package_section = True
            continue
        
        # Extract ULD data
        if uld_section and uld_pattern.match(line):
            uld_id, length, width, height, weight_limit = line.split(',')
            uld_data.append({
                "ULD Identifier": uld_id,
                "Length (cm)": int(length),
                "Width (cm)": int(width),
                "Height (cm)": int(height),
                "Weight Limit (kg)": int(weight_limit)
            })
        
        # Extract package data
        if package_section and package_pattern.match(line):
            parts = line.split(',')
            package_data.append({
                "Package Identifier": parts[0],
                "Length (cm)": int(parts[1]),
                "Width (cm)": int(parts[2]),
                "Height (cm)": int(parts[3]),
                "Weight (kg)": int(parts[4]),
                "Type": parts[5],
                "Cost of Delay": parts[6] if parts[6] != '-' else None
            })
    
    return uld_data, package_data


# Function to save the solution to a text file
def save_solution_to_txt(ans, total_cost, num_priority_ULDs, output_file):
    """
    Converts a solution DataFrame to the specified .txt format.

    Args:
        ans (pd.DataFrame): The solution DataFrame with columns:
            - Package-ID
            - ULD Identifier
            - x0, y0, z0, x1, y1, z1
        total_cost (int): The total cost of the solution.
        num_priority_ULDs (int): The number of ULDs with priority packages.
        output_file (str): Path to the output .txt file.
    """
    # Count total number of packages packed in ULDs
    total_packed_packages = ans[ans["ULD Identifier"] != "None"].shape[0]

    # Write the first line with total cost, total packages, and number of priority ULDs
    with open(output_file, 'w') as f:
        f.write(f"{total_cost},{total_packed_packages},{num_priority_ULDs}\n")

        # Iterate over the DataFrame and write package details
        for _, row in ans.iterrows():
            package_id = row["Package Identifier"]
            uld_id = row["ULD Identifier"]
            x0, y0, z0 = row["X0"], row["Y0"], row["Z0"]
            x1, y1, z1 = row["X1"], row["Y1"], row["Z1"]

            if uld_id == "None":
                uld_id = "NONE"
                x0, y0, z0, x1, y1, z1 = -1, -1, -1, -1, -1, -1

            f.write(f"{package_id},{uld_id},{x0},{y0},{z0},{x1},{y1},{z1}\n")


# Function to calculate package volume
def calculate_volume(x0, y0, z0, x1, y1, z1):
    return max(0, x1 - x0) * max(0, y1 - y0) * max(0, z1 - z0)


# Usage
file_path = 'Challange_FedEx.txt'  # Replace with your text file's path
uld_data, package_data = extract_data(file_path)
pkgs_df = pd.DataFrame(package_data)
uld_df = pd.DataFrame(uld_data)
pkgs_df['Cost of Delay'] = pd.to_numeric(pkgs_df['Cost of Delay'], errors='coerce')
pkgs_df['Cost of Delay'] = pd.to_numeric(pkgs_df['Cost of Delay'], errors='coerce')
pkgs_df["Volume (cm3)"] = pkgs_df["Length (cm)"] * pkgs_df["Width (cm)"] * pkgs_df["Height (cm)"]
uld_df["Volume (cm3)"] = uld_df["Length (cm)"] * uld_df["Width (cm)"] * uld_df["Height (cm)"]


total_pri_vol = pkgs_df[pkgs_df["Type"] == "Priority"]["Volume (cm3)"].sum()
total_pri_weight = pkgs_df[pkgs_df["Type"] == "Priority"]["Weight (kg)"].sum()
uld_df = uld_df.sort_values(by="Volume (cm3)", ascending=False)
totvol = 0
for i in range(uld_df.shape[0]):
    totvol += uld_df.iloc[i]["Volume (cm3)"]
    if totvol >= total_pri_vol:
        break

vol_min = i+1

uld_df = uld_df.sort_values(by="Weight Limit (kg)", ascending=False)
totwt = 0
for i in range(uld_df.shape[0]):
    totwt += uld_df.iloc[i]["Weight Limit (kg)"]
    if totwt >= total_pri_weight:
        break

wt_min = i+1

min_req = max(vol_min, wt_min)

pri_pkgs_df_vol = pkgs_df[pkgs_df["Type"] == "Priority"].sort_values(by=["Volume (cm3)", "Package Identifier"], ascending=False).reset_index(drop=True) 
uld_df = uld_df.sort_values(by=["Volume (cm3)", "ULD Identifier"], ascending=False).reset_index(drop=True)
uld_list = uld_df[['ULD Identifier','Length (cm)', 'Width (cm)', 'Height (cm)', 'Weight Limit (kg)']].values.tolist()
pri_boxes_vol = [Box(pri_pkgs_df_vol.loc[i, 'Package Identifier'], pri_pkgs_df_vol.loc[i, 'Length (cm)'], pri_pkgs_df_vol.loc[i, 'Width (cm)'], pri_pkgs_df_vol.loc[i, 'Height (cm)'], pri_pkgs_df_vol.loc[i, 'Weight (kg)'], True) for i in range(pri_pkgs_df_vol.shape[0])]
population = [[[i, randint(0, 5)] for i in range(len(pri_boxes_vol))] for _ in range(500)]
ulds = uld_list[:min_req]
print("Genetic Algorithm is running for part 1... \n")

#print(f"Population:50, Generations:100, pm1 = 0.2, pm2= 0.01")
new_pop, best_gene, best_fitness, fit_list =genetic_algorithm(ulds, pri_boxes_vol, 50, 100, 0.01, 0.01, initial_pop=population)

ans_df = ans_from_gene(uld_list[:min_req+1], pri_boxes_vol, best_gene)



eco_pkgs_df = pkgs_df[pkgs_df['Type']=='Economy'].sort_values(by=["Cost of Delay", "Package Identifier"], ascending=False).reset_index(drop=True)
eco_boxes = [Box(eco_pkgs_df.loc[i, 'Package Identifier'], eco_pkgs_df.loc[i, 'Length (cm)'], eco_pkgs_df.loc[i, 'Width (cm)'], eco_pkgs_df.loc[i, 'Height (cm)'], eco_pkgs_df.loc[i, 'Weight (kg)'], False, cost=eco_pkgs_df.loc[i, "Cost of Delay"]) for i in range(eco_pkgs_df.shape[0])]
pri_boxes = [Box(pri_pkgs_df_vol.loc[i, 'Package Identifier'], pri_pkgs_df_vol.loc[i, 'Length (cm)'], pri_pkgs_df_vol.loc[i, 'Width (cm)'], pri_pkgs_df_vol.loc[i, 'Height (cm)'], pri_pkgs_df_vol.loc[i, 'Weight (kg)'], True) for i in range(pri_pkgs_df_vol.shape[0])]
all_boxes = pri_boxes + eco_boxes


pri_population = new_pop
eco_population = [[[i+len(pri_boxes), randint(0, 5)] for i in range(len(eco_boxes))] for _ in range(len(pri_population))]
population = [pri_population[i] + eco_population[i] for i in range(len(pri_population))]
print("Genetic Algorithm is running for part 2... \n")
# print(f"Population:50, Generations :200, pm1 = 0.2, pm2= 0.01")
full_pop, best_full_gene, best_full_fitness, fit_list =genetic_algorithm(uld_list, all_boxes, 50, 100, 0.2, 0.01, initial_pop=population, cost="normal", k=5000, a=len(pri_boxes))


ans_df = ans_from_gene(uld_list, all_boxes, best_full_gene, cost_type="normal")
np.save('best_gene_final_all.npy', best_full_gene)  # Saving the best file


# Replace NaN values in x, y, z with -1
best_full_gene = np.load('best_gene_final_all.npy')
ans = ans_from_gene(uld_list, all_boxes, best_full_gene, cost_type="normal").fillna(-1)

report = validity_report(uld_df, pkgs_df, ans, k=5000)



best_full_gene = np.load('best_gene_final_all.npy')

# Compute ans using the provided function ans_from_gene
ans = ans_from_gene(uld_list, all_boxes, best_full_gene, cost_type="normal").fillna(-1)

# Assume total_cost and num_priority_ULDs are computed from the solution
total_cost = report['cost']  # Example value
num_priority_ULDs = 3  # Example value

# Save to a .txt file
output_file = 'solution.txt'
save_solution_to_txt(ans, total_cost, num_priority_ULDs, output_file)

#Visualization Part

# Read the file
file_path = "solution.txt"  # Replace with your actual file path
uld_packages = {}

with open(file_path, "r") as file:
    lines = file.readlines()[1:]  # Skip the header

# Parse the file
for line in lines:
    parts = line.strip().split(",")
    package_id, uld, *coords = parts
    coords = list(map(float, coords))
    if uld != "NONE":
        volume = calculate_volume(*coords)
        if uld not in uld_packages:
            uld_packages[uld] = []
        uld_packages[uld].append(volume)

# Calculate volumetric efficiency
results = {}
efficiencies = []  # List to store individual efficiencies
for uld, package_volumes in uld_packages.items():
    total_package_volume = sum(package_volumes)
    uld_dimensions = uld_df[uld_df["ULD Identifier"] == uld]
    if not uld_dimensions.empty:
        uld_volume = (
            uld_dimensions["Length (cm)"].iloc[0]
            * uld_dimensions["Width (cm)"].iloc[0]
            * uld_dimensions["Height (cm)"].iloc[0]
        )
        efficiency = (total_package_volume / uld_volume) * 100
        results[uld] = efficiency
        efficiencies.append(efficiency)

# Calculate mean volumetric efficiency
if efficiencies:
    mean_efficiency = sum(efficiencies) / len(efficiencies)
else:
    mean_efficiency = 0

# Print results
sorted_results = dict(sorted(results.items()))


for uld, efficiency in results.items():
    print(f"ULD {uld} Volumetric Efficiency: {efficiency:.2f}%")
print(f"Mean Volumetric Efficiency: {mean_efficiency:.2f}%")


visualize(uld_df, pkgs_df, ans,sorted_results)

print(f"Solution saved to {output_file}")