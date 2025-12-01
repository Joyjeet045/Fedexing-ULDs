import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def packs(uld_dims: list, pack_location: list) -> bool:
    if any([pack_location[i]<0 for i in range(6)]):
        return False
    if any([pack_location[i]>uld_dims[i%3] for i in range(6)]):
        return False
    return True

def overlap(pack1: list, pack2: list) -> bool:
    # Check if there's overlap in all three dimensions
    return (
        pack1[3] > pack2[0] and pack1[0] < pack2[3] and  # x-dimension
        pack1[4] > pack2[1] and pack1[1] < pack2[4] and  # y-dimension
        pack1[5] > pack2[2] and pack1[2] < pack2[5]      # z-dimension
    )

def support(box1: tuple[int, int, int, int, int, int], box2: tuple[int, int, int, int, int, int]) -> bool:
        # Check if box2 is supported by box1
        return (box1[0] <= box2[0] and box1[1] <= box2[1] and box1[3] >= box2[3] and box1[4] >= box2[4])

def support_area(box1: tuple[int, int, int, int, int, int], box2: tuple[int, int, int, int, int, int]) -> int:
        # Calculate the support area of box2 on box1
        dx = min(box1[3], box2[3]) - max(box1[0], box2[0])
        dy = min(box1[4], box2[4]) - max(box1[1], box2[1])
        box2_area = (box2[3] - box2[0]) * (box2[4] - box2[1])
        return dx*dy/box2_area if dx > 0 and dy > 0 else 0


def cost(ULD_list: pd.DataFrame, package_list: pd.DataFrame, placement_list: pd.DataFrame, k: int = 100) -> int:
    """
    Calculate the total cost of delay for all packages.

    Parameters:
    - ULD_list: DataFrame with ULD details.
    - package_list: DataFrame with package details.
    - placement_list: DataFrame with package placements in ULDs.
    - k: Cost of each ULD containing a priority package.

    Returns:
    - Total cost of delay for all packages.
    """
    total_cost = 0
    uld_list = ULD_list['ULD Identifier'].values
    priority_ulds = set()
    for _, row in placement_list.iterrows():
        package_id = row['Package Identifier']
        uld_id = row['ULD Identifier']
        if (uld_id not in uld_list):
            total_cost += package_list[package_list['Package Identifier'] == package_id]['Cost of Delay'].values[0]
            continue
        uld_length = ULD_list[ULD_list['ULD Identifier'] == row['ULD Identifier']]['Length (cm)'].values[0]
        uld_width = ULD_list[ULD_list['ULD Identifier'] == row['ULD Identifier']]['Width (cm)'].values[0]
        uld_height = ULD_list[ULD_list['ULD Identifier'] == row['ULD Identifier']]['Height (cm)'].values[0]
        if (not packs([uld_length, uld_width, uld_height], [row['X0'], row['Y0'], row['Z0'], row['X1'], row['Y1'], row['Z1']])):
            total_cost += package_list[package_list['Package Identifier'] == package_id]['Cost of Delay'].values[0]
        if package_list[package_list['Package Identifier'] == package_id]['Type'].values[0] == 'Priority':
            priority_ulds.add(row['ULD Identifier'])
        

        
        
    total_cost += k * len(priority_ulds)
            
    return total_cost

def validity_report(ULD_list: pd.DataFrame, package_list: pd.DataFrame, placement_list: pd.DataFrame, k:int = 100)-> dict:
    # ULD_list: a dataframe with columns ['ULD Identifier', 'Length (cm)', 'Width (cm)', 'Height (cm)', 'Weight Limit (kg)']
    # eg: ULD_list = pd.DataFrame({'ULD Identifier': ['ULD1', 'ULD2'], 'Length': [100, 200], 'Width': [100, 200], 'Height': [100, 200], 'Weight Limit': [1000, 2000]})
    # package_list: a dataframe with columns ['Package Identifier', 'Length (cm)', 'Width (cm)', 'Height (cm)', 'Weight (kg)', 'Type', 'Cost of Delay']
    # eg: package_list = pd.DataFrame({'Package Identifier': ['P1', 'P2'], 'Length': [10, 20], 'Width': [10, 20], 'Height': [10, 20], 'Weight': [10, 20], 'Type': ['P', 'E'], 'Cost of Delay': [None, 200]})
    # placement_list: a dataframe with columns ['Package Identifier', 'ULD Identifier', 'X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1']
    # X0, Y0, Z0: the coordinates of the bottom left corner of the package
    # X1, Y1, Z1: the coordinates of the top right corner of the package
    # eg: placement_list = pd.DataFrame({'Package Identifier': ['P1', 'P2'], 'ULD Identifier': ['ULD1', 'ULD2'], 'X0': [0, 0], 'Y0': [0, 0], 'Z0': [0, 0], 'X1': [10, 20], 'Y1': [10, 20], 'Z1': [10, 20]})
    # k = cost of each ULD containing a priority package

    # return: a dictionary with the following keys: "ULD Exceeding Weight Limit", "Package Overlapping", "Package Outside ULD"

    report = {"Weight_limit": [], "Overlapping": [], "Outside": [], "PriorityULDs": 0, "cost": 0}
    # Check if any packages are not assigned to a ULD
    for i in range(len(placement_list)):
        package = placement_list.iloc[i]
        if package['ULD Identifier'] not in ULD_list['ULD Identifier'].values:
            report["Outside"].append((package['Package Identifier'], package['ULD Identifier']))
    # Check conditions for each ULD
    for i in range(len(ULD_list)):
        ULD = ULD_list.iloc[i]
        ULD_identifier = ULD['ULD Identifier']
        ULD_length = ULD['Length (cm)']
        ULD_width = ULD['Width (cm)']
        ULD_height = ULD['Height (cm)']
        ULD_weight_limit = ULD['Weight Limit (kg)']
        # Check if the ULD is empty
        if placement_list[placement_list['ULD Identifier'] == ULD_identifier].shape[0] == 0:
            continue
        # Check if the weight limit is exceeded
        #assigned_weight = placement_list[placement_list['ULD Identifier'] == ULD_identifier]['Weight (kg)'].sum()
        assigned_weight = 0
        for pack in placement_list[placement_list['ULD Identifier'] == ULD_identifier].iterrows():
            assigned_weight += package_list[package_list['Package Identifier'] == pack[1]['Package Identifier']]['Weight (kg)'].values[0]
        
        if assigned_weight > ULD_weight_limit:
            report["Weight_limit"].append((ULD_identifier, ULD_weight_limit, assigned_weight)) 
        # Check if the ULD assigned packages are within the ULD
        for pack in placement_list[placement_list['ULD Identifier'] == ULD_identifier].iterrows():
            package = pack[1]
            pack_location = [package['X0'], package['Y0'], package['Z0'], package['X1'], package['Y1'], package['Z1']]
            if not packs([ULD_length, ULD_width, ULD_height], pack_location):      
                report["Outside"].append((package['Package Identifier'], ULD_identifier))
        
        # Check if the packages are overlapping
        for pack1 in placement_list[placement_list['ULD Identifier'] == ULD_identifier].iterrows():
            for pack2 in placement_list[placement_list['ULD Identifier'] == ULD_identifier].iterrows():
                # print(pack1[0], pack2[0])
                if (pack2[0]) > (pack1[0]):
                    pack_1_loc = [pack1[1]['X0'], pack1[1]['Y0'], pack1[1]['Z0'], pack1[1]['X1'], pack1[1]['Y1'], pack1[1]['Z1']]
                    pack_2_loc = [pack2[1]['X0'], pack2[1]['Y0'], pack2[1]['Z0'], pack2[1]['X1'], pack2[1]['Y1'], pack2[1]['Z1']]
                    if overlap(pack_1_loc, pack_2_loc):
                        report["Overlapping"].append((pack1[1]['Package Identifier'], pack2[1]['Package Identifier'], ULD_identifier))
    # Check the number of ULDs with priority packages
    for i in range(len(ULD_list)):
        ULD = ULD_list.iloc[i]
        ULD_identifier = ULD['ULD Identifier']
        for pack in placement_list[placement_list['ULD Identifier'] == ULD_identifier].iterrows():
            package = pack[1]
            if package_list[package_list['Package Identifier'] == package['Package Identifier']]['Type'].values[0] == 'Priority':
                report["PriorityULDs"] += 1
                break
    
    tot_cost = cost(ULD_list, package_list, placement_list, k)
    report["cost"] = tot_cost
    return report
        
def visualize(ULD_list: pd.DataFrame, package_list: pd.DataFrame, placement_list: pd.DataFrame, efficiencies: dict):
    """
    Visualizes ULDs and packages in a 3D space with volumetric efficiencies.

    Parameters:
    - ULD_list: DataFrame with ULD details.
    - package_list: DataFrame with package details.
    - placement_list: DataFrame with package placements in ULDs.
    - efficiencies: Dictionary with ULD identifiers as keys and their volumetric efficiencies as values.
    """
    def draw_cuboid(ax, x0, y0, z0, x1, y1, z1, color, label=None, alpha=0.5):
        """Helper function to draw a cuboid in 3D."""
        # Vertices of the cuboid
        vertices = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # Bottom face
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # Top face
        ]
        # Define the 6 faces using the vertices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        ]
        # Add the cuboid to the plot
        poly3d = Poly3DCollection(faces, alpha=alpha, linewidths=1, edgecolors='black')
        poly3d.set_facecolor(color)
        ax.add_collection3d(poly3d)
        # Add a label at the center of the cuboid
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, label, color='black')

    # Initialize the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title('ULD and Package Visualization with Efficiencies')

    # Draw ULDs
    colors = ['cyan', 'orange', 'lightgreen', 'pink']  # Colors for ULDs
    offset_dict = {}  # Dictionary to store the offset for each ULD
    for i, row in ULD_list.iterrows():
        # Calculate the offset for the ULD
        x_offset = sum(ULD_list['Length (cm)'][:i])  # Offset based on cumulative lengths of preceding ULDs
        offset_dict[row['ULD Identifier']] = x_offset
        # Define the cuboid for the ULD
        x0, y0, z0 = x_offset, 0, 0
        x1, y1, z1 = x_offset + row['Length (cm)'], row['Width (cm)'], row['Height (cm)']
        
        # Draw the ULD cuboid
        draw_cuboid(ax, x0, y0, z0, x1, y1, z1, colors[i % len(colors)], label=row['ULD Identifier'], alpha=0.3)
        
        # Add efficiency text above the ULD
        efficiency = efficiencies.get(row['ULD Identifier'], 0)
        ax.text((x0 + x1) / 2, y1 + 100, (z0 + z1) / 2 + z1, f"{efficiency:.2f}%", color='blue', fontsize=10, fontweight='bold')

    # Draw packages
    for _, row in placement_list.iterrows():
        if row['ULD Identifier'] not in offset_dict:
            continue
        package_id = row['Package Identifier']
        x0, y0, z0 = row['X0'], row['Y0'], row['Z0']
        x1, y1, z1 = row['X1'], row['Y1'], row['Z1']
        color = 'blue' if package_list[package_list['Package Identifier'] == package_id]['Type'].values[0] == 'Priority' else 'red'
        uld_id = row['ULD Identifier']
        x_offset = offset_dict[uld_id]
        x0 += x_offset
        x1 += x_offset
        draw_cuboid(ax, x0, y0, z0, x1, y1, z1, color, label=package_id, alpha=0.7)

    # Adjust the view and show the plot
    max_length = sum(ULD_list['Length (cm)'])
    max_width = ULD_list['Width (cm)'].max()
    max_height = ULD_list['Height (cm)'].max()
    ax.set_box_aspect([max_length, max_width, max_height])  # Adjust aspect ratio
    ax.set_xlim(0, max_length)
    ax.set_ylim(0, max_width)
    ax.set_zlim(0, max_height)
    plt.show()


# Orders packages with all priority packages in front (random order)
# followed by economy packages ordered according to cost/volume and cost/weight
def order(df_ULD: pd.DataFrame, df_packages: pd.DataFrame, ULD_volume = 'Volume (cm3)', packages_volume= 'Volume (cm3)'):
    #df_packages.loc[df_packages['Type']=='Priority', 'Cost of Delay']=np.inf
    df_priority = df_packages[df_packages['Type']=='Priority']   
    df_economy = df_packages[df_packages['Type']=='Economy']
    
    df_packages['Cost/Volume']=df_packages['Cost of Delay']/df_packages[packages_volume]
    df_packages['Cost/Weight']=df_packages['Cost of Delay']/df_packages['Weight (kg)']
    
    #Normalize
    df_packages['Cost/Volume'] = (df_packages['Cost/Volume']-df_packages['Cost/Volume'].min(skipna=True))/(df_packages['Cost/Volume'].max(skipna=True)-df_packages['Cost/Volume'].min(skipna=True))
    df_packages['Cost/Weight'] = (df_packages['Cost/Weight']-df_packages['Cost/Weight'].min(skipna=True))/(df_packages['Cost/Weight'].max(skipna=True)-df_packages['Cost/Weight'].min(skipna=True))
    
    #Volume and Weight available for economy
    economy_volume = df_ULD[ULD_volume].sum()-df_priority[packages_volume].sum()
    economy_weight = df_ULD['Weight Limit (kg)'].sum()-df_priority['Weight (kg)'].sum()
    
    volume_ratio = (df_economy['Volume (cm3)'].sum()-economy_volume)/economy_volume
    weight_ratio = (df_economy['Weight (kg)'].sum() - economy_weight)/economy_weight

    if volume_ratio<0:volume_ratio=0
    if weight_ratio<0:weight_ratio=0

    volume_factor = volume_ratio/(volume_ratio+weight_ratio)
    weight_factor = weight_ratio/(volume_ratio+weight_ratio)

    df_packages['Priority Factor'] = volume_factor*df_packages['Cost/Volume']+weight_factor*df_packages['Cost/Weight']
    df_ordered = df_packages.sort_values(by=['Type', 'Priority Factor'], ascending=[False,False]).reset_index(drop=True)

    return df_ordered


def vol_stack(df:pd.DataFrame, length_limit, width_limit, height_limit, weight_limit):
    df = df.copy()
    unplaced_df = df[df['quantity']>0]
    max_volume = 0
    df_stacked=pd.DataFrame()
    for idx, package in unplaced_df[unplaced_df['Type']=='Priority'].iterrows():
        length = package['Length (cm)']
        width = package['Width (cm)']
        height = package['Height (cm)']
        weight = package['Weight (kg)']

        if length <= length_limit and width <= width_limit and height <= height_limit and weight<weight_limit:
            # Compute the volume
            volume = length*width*height
            df.loc[idx,'quantity']=0
            volume_ontop, df_stacked_ontop = vol_stack(df,length,width,height_limit-height,weight_limit-weight)
            volume+=volume_ontop
            if volume>max_volume:
                max_volume = volume
                df_stacked = pd.concat([df.loc[[idx]],df_stacked_ontop])

    return max_volume, df_stacked



def stack(df_packages: pd.DataFrame, current_stack: list[pd.Series], height_limit, weight_limit, uid):
    base_length = current_stack[-1]['Length (cm)']
    base_width = current_stack[-1]['Width (cm)']

    stack_factor = np.inf  # Initialize stack factor
    best_stack_idx = None

    unplaced_df = df_packages[df_packages['quantity']>0]
    # Find the best package to place
    for idx, package in unplaced_df[unplaced_df['Type']=='Priority'].iterrows():
        length = package['Length (cm)']
        width = package['Width (cm)']
        height = package['Height (cm)']
        weight = package['Weight (kg)']

        if length <= base_length and width <= base_width and height <= height_limit and weight<weight_limit:
            # Compute the stack factor
            new_stack_factor = base_length * (base_width - width) + base_width * (base_length - length)
            if new_stack_factor < stack_factor:
                stack_factor = new_stack_factor
                best_stack_idx = idx

    # No priority can fit. Check economy
    if best_stack_idx is None:
        for idx, package in unplaced_df[unplaced_df['Type']=='Economy'].iterrows():
            length = package['Length (cm)']
            width = package['Width (cm)']
            height = package['Height (cm)']
            weight = package['Weight (kg)']

            if length <= base_length and width <= base_width and height <= height_limit and weight<weight_limit:
                #This is the best cost/vol wt package which can fit
                best_stack_idx = idx
                break

    # Base case: No more packages can fit
    if best_stack_idx is None:
        return current_stack

    # Place the best package
    df_packages.loc[best_stack_idx, ['ULD Identifier','X0', 'Y0', 'Z0', 'quantity']] = [
        uid,
        current_stack[-1]['X0'],
        current_stack[-1]['Y0'],
        current_stack[-1]['Z0'] + current_stack[-1]['Height (cm)'],
        0
    ]

    current_stack.append(df_packages.loc[best_stack_idx])


    # Update base dimensions and height limit for the next package
    new_height_limit = height_limit - df_packages.loc[best_stack_idx]['Height (cm)']
    new_weight_limit = weight_limit - df_packages.loc[best_stack_idx]['Weight (kg)']
    # Continue stacking recursively
    return stack(df_packages, current_stack, new_height_limit, new_weight_limit, uid)



