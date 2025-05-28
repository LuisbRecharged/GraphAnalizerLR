from graphviz import Digraph, Graph
from PIL import Image
import matplotlib.pyplot as plt
import copy
import random
import heapq
#Universidad de Guanajuato DICIS - Discrete Math / Linear Algebra Final Proyect.
#Graph properties and algorithms analizer.
#LB.
def symmetrize_adjacency_matrix(directed_adj_matrix):
    n = len(directed_adj_matrix)
    sym_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if directed_adj_matrix[i][j] == 1 or directed_adj_matrix[j][i] == 1:
                sym_matrix[i][j] = 1
    return sym_matrix

def symmetrize_weight_matrix(directed_adj_matrix, directed_weight_matrix):
    if not directed_weight_matrix:
        return None
    n = len(directed_adj_matrix)
    sym_weights = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            edge_exists_undirected = directed_adj_matrix[i][j] == 1 or directed_adj_matrix[j][i] == 1
            if edge_exists_undirected:
                w_ij = directed_weight_matrix[i][j] if directed_adj_matrix[i][j] == 1 and directed_weight_matrix[i][j] > 0 else 0
                w_ji = directed_weight_matrix[j][i] if directed_adj_matrix[j][i] == 1 and directed_weight_matrix[j][i] > 0 else 0
                chosen_weight = 0
                if w_ij > 0 and w_ji > 0: chosen_weight = min(w_ij, w_ji)
                elif w_ij > 0: chosen_weight = w_ij
                elif w_ji > 0: chosen_weight = w_ji
                if chosen_weight > 0:
                    sym_weights[i][j] = chosen_weight
                    sym_weights[j][i] = chosen_weight
    return sym_weights

def calculate_degrees(matrix):
    n = len(matrix)
    degrees = []
    for i in range(n):
        out_degree = sum(matrix[i])
        in_degree = sum(matrix[j][i] for j in range(n))
        degrees.append((in_degree, out_degree))
    return degrees

def calculate_undirected_degrees(undirected_matrix):
    n = len(undirected_matrix)
    degrees = []
    for i in range(n):
        degrees.append(sum(undirected_matrix[i])) 
    return degrees

def is_reflexive(matrix):
    return all(matrix[i][i] == 1 for i in range(len(matrix)))

def is_symmetric(matrix):
    n = len(matrix)
    return all(matrix[i][j] == matrix[j][i] for i in range(n) for j in range(n))

def is_antisymmetric(matrix):
    n = len(matrix)
    return not any(i != j and matrix[i][j] == 1 and matrix[j][i] == 1 for i in range(n) for j in range(n))

def is_transitive(matrix):
    n = len(matrix)
    return all(not (matrix[i][j] and matrix[j][k] and not matrix[i][k]) for i in range(n) for j in range(n) for k in range(n))

def find_cycle(adj_matrix): # Finds the shortest cycle using BFS
    n = len(adj_matrix)
    shortest_cycle = None
    for start_node_bfs in range(n):
        # visited stores: {node_index: (distance_from_start, path_from_start)}
        visited_bfs = {start_node_bfs: (0, [start_node_bfs])} 
        queue_bfs = [(start_node_bfs, [start_node_bfs])] # (current_node, path_to_current_node)
        head = 0
        while head < len(queue_bfs):
            current_node, current_path = queue_bfs[head]; head += 1
            for neighbor_node in range(n):
                if adj_matrix[current_node][neighbor_node]: # If there's an edge
                    if neighbor_node == start_node_bfs: # Found a cycle back to the start_node_bfs
                        if len(current_path) > 2 : 
                            formed_cycle = current_path + [start_node_bfs]
                            if not shortest_cycle or len(formed_cycle) < len(shortest_cycle):
                                shortest_cycle = formed_cycle
                    elif neighbor_node not in visited_bfs: 
                        visited_bfs[neighbor_node] = (len(current_path), current_path + [neighbor_node])
                        queue_bfs.append((neighbor_node, current_path + [neighbor_node]))
    return shortest_cycle

def graph_coloring(adj_matrix):
    n = len(adj_matrix)
    colors = [-1] * n
    if n == 0: return colors
    symmetric_matrix_for_coloring = [[max(adj_matrix[i][j], adj_matrix[j][i]) for j in range(n)] for i in range(n)]
    
    for node_to_color in range(n):
        used_neighbor_colors = set()
        for neighbor in range(n):
            if node_to_color == neighbor: continue 
            if symmetric_matrix_for_coloring[node_to_color][neighbor] and colors[neighbor] != -1:
                used_neighbor_colors.add(colors[neighbor])
        assigned_color = 0
        while assigned_color in used_neighbor_colors: assigned_color += 1
        colors[node_to_color] = assigned_color
    return colors

def draw_basic_graph(directed_adj_matrix, relation_type, directed_degrees, directed_weight_matrix=None):
    dot = Digraph(name='basic_directed_graph')
    dot.graph_attr.update({'splines': 'spline', 'ranksep': '0.75', 'nodesep': '0.5', 'concentrate': 'true'})
    dot.node_attr.update({'penwidth': '1.0', 'fontname': 'Arial'})
    dot.edge_attr.update({'penwidth': '0.8', 'fontname': 'Arial'})

    n = len(directed_adj_matrix)
    nodes = [chr(97 + i) for i in range(n)]
    for i, node in enumerate(nodes):
        in_degree, out_degree = directed_degrees[i]
        label = f"{node}\nEnt: {in_degree}\nSal: {out_degree}"
        dot.node(node, label=label, shape='ellipse')
    edge_styles = {"equivalencia": {'dir': 'none', 'color': 'green'}, "orden parcial": {'color': 'red'}, "general": {}}
    style = edge_styles.get(relation_type, {})
    for i in range(n):
        for j in range(n):
            if directed_adj_matrix[i][j]:
                edge_style = style.copy()
                if directed_weight_matrix and directed_weight_matrix[i][j] > 0 :
                    edge_style['label'] = str(directed_weight_matrix[i][j])
                if relation_type == "equivalencia":
                    if i == j: dot.edge(nodes[i], nodes[j], dir='none', color='blue', **edge_style)
                    elif directed_adj_matrix[j][i] and i < j: dot.edge(nodes[i], nodes[j], **edge_style) 
                    elif not directed_adj_matrix[j][i]: dot.edge(nodes[i], nodes[j], **edge_style)
                else:
                    if i == j: dot.edge(nodes[i], nodes[j], color='blue', **edge_style) 
                    else: dot.edge(nodes[i], nodes[j], **edge_style)
    title = f"Grafo Original (Dirigido)\nTipo: {relation_type}, {n}x{n}\nGrados Ent/Sal"
    if directed_weight_matrix: title += "\n(Pesos Dirigidos)"
    dot.attr(label=title, labelloc='t', fontsize='14', fontname='Arial Bold')
    return dot

def draw_colored_graph(undirected_adj_matrix, colors, undirected_degrees_val, cycle_data, undirected_weight_matrix=None):
    dot = Graph(name='colored_undirected_graph', engine='neato') 
    dot.graph_attr.update({'splines': 'true', 'overlap': 'scale', 'sep': '0.6', 'K': '0.8'}) 
    dot.node_attr.update({'penwidth': '1.0', 'fontname': 'Arial'})
    dot.edge_attr.update({'penwidth': '0.8', 'fontname': 'Arial'})

    n = len(undirected_adj_matrix)
    nodes = [chr(97 + i) for i in range(n)]
    color_palette = ["#FF9999", "#99FF99", "#9999FF", "#FFFF99", "#FFCC99", "#99FFFF", "#FF99FF", "#CCFF99", "#99CCFF", "#FFCCFF", "#CCCCCC", "#C0C0C0", "#A0A0A0", "#808080", "#606060"]

    for i, node in enumerate(nodes):
        color_idx = colors[i] % len(color_palette) if colors[i] != -1 else (len(color_palette) - 1) 
        color_hex = color_palette[color_idx]
        color_name = f"Color {colors[i] + 1}" if colors[i] != -1 else "Sin color"
        degree_val = undirected_degrees_val[i]
        label = f"{node}\n{color_name}\nGrado: {degree_val}" 
        dot.node(node, label=label, style='filled', fillcolor=color_hex, fontcolor='black', shape='ellipse')

    for i in range(n):
        for j in range(i, n): 
            if undirected_adj_matrix[i][j]:
                u, v = nodes[i], nodes[j]
                is_cycle_edge = False
                if cycle_data:
                    current_edge_sorted_tuple = tuple(sorted((i,j)))
                    for k_cycle in range(len(cycle_data) - 1):
                        cycle_segment_tuple = tuple(sorted((cycle_data[k_cycle], cycle_data[k_cycle+1])))
                        if current_edge_sorted_tuple == cycle_segment_tuple:
                            is_cycle_edge = True
                            break
                
                if not is_cycle_edge: 
                    edge_attrs = {}
                    if undirected_weight_matrix and undirected_weight_matrix[i][j] > 0:
                        edge_attrs['label'] = str(undirected_weight_matrix[i][j])
                    if i == j: 
                        edge_attrs['color'] = 'blue' 
                    dot.edge(u,v, **edge_attrs)

    if cycle_data:
        for k in range(len(cycle_data) - 1):
            u_idx, v_idx = cycle_data[k], cycle_data[k+1]
            u_char, v_char = nodes[u_idx], nodes[v_idx]
            edge_attrs_cycle = {'color': 'red', 'penwidth': '2.5'}
            if undirected_weight_matrix and undirected_weight_matrix[u_idx][v_idx] > 0:
                edge_attrs_cycle['label'] = str(undirected_weight_matrix[u_idx][v_idx])
            dot.edge(u_char, v_char, **edge_attrs_cycle)

    num_colors_used = max(colors) + 1 if any(c != -1 for c in colors) else 0
    title = f"Coloreado (Grafo No Dirigido)\nColores usados: {num_colors_used}"
    if cycle_data: title += f"\nCiclo ({len(cycle_data)-1} aristas) en rojo"
    dot.attr(label=title, labelloc='t', fontsize='14', fontname='Arial Bold')
    return dot

def has_euler_path(undirected_adj_matrix):
    n = len(undirected_adj_matrix);
    if n == 0: return False, None 
    degrees = calculate_undirected_degrees(undirected_adj_matrix)
    odd_degree_count = sum(1 for d in degrees if d % 2 != 0)

    if not is_connected_for_euler(undirected_adj_matrix): return False, None

    if odd_degree_count == 0: return True, "cycle" 
    elif odd_degree_count == 2: return True, "path"  
    else: return False, None

def is_connected_for_euler(undirected_adj_matrix):
    n = len(undirected_adj_matrix);
    if n == 0: return True 
    
    nodes_with_edges = set()
    has_any_edge = False
    for r_node in range(n):
        for c_node in range(r_node, n): 
            if undirected_adj_matrix[r_node][c_node] == 1:
                has_any_edge = True
                nodes_with_edges.add(r_node)
                if r_node != c_node : nodes_with_edges.add(c_node) 
    
    if not has_any_edge: return True # All nodes isolated, or single node

    start_node = next(iter(nodes_with_edges)) if nodes_with_edges else 0
    
    visited_bfs = [False] * n; queue_bfs = [start_node]; visited_bfs[start_node] = True; head_bfs = 0
    count_visited_in_component = 0
    
    while head_bfs < len(queue_bfs):
        u_bfs = queue_bfs[head_bfs]; head_bfs += 1
        if u_bfs in nodes_with_edges: 
            count_visited_in_component +=1
            
        for v_neighbor_bfs in range(n):
            if undirected_adj_matrix[u_bfs][v_neighbor_bfs] and not visited_bfs[v_neighbor_bfs]:
                visited_bfs[v_neighbor_bfs] = True; queue_bfs.append(v_neighbor_bfs)
    
    return count_visited_in_component == len(nodes_with_edges)

def is_strongly_connected(directed_matrix): 
    n = len(directed_matrix)
    if n == 0: return True
    for start_node_check in range(n):
        visited_dfs = [False]*n; temp_stack = [start_node_check]; visited_dfs[start_node_check] = True; nodes_reached_count = 0
        while temp_stack:
            u = temp_stack.pop(); nodes_reached_count += 1
            for v_neighbor in range(n):
                if directed_matrix[u][v_neighbor] and not visited_dfs[v_neighbor]:
                    visited_dfs[v_neighbor] = True; temp_stack.append(v_neighbor)
        if nodes_reached_count < n : return False 

    transposed_matrix = [[directed_matrix[j][i] for j in range(n)] for i in range(n)]
    for start_node_check_rev in range(n):
        visited_dfs_rev = [False]*n; temp_stack_rev = [start_node_check_rev]; visited_dfs_rev[start_node_check_rev] = True; nodes_reached_count_rev = 0
        while temp_stack_rev:
            u = temp_stack_rev.pop(); nodes_reached_count_rev +=1
            for v_neighbor in range(n):
                if transposed_matrix[u][v_neighbor] and not visited_dfs_rev[v_neighbor]:
                    visited_dfs_rev[v_neighbor] = True; temp_stack_rev.append(v_neighbor)
        if nodes_reached_count_rev < n : return False
    return True

def is_connected(undirected_adj_matrix): 
    n = len(undirected_adj_matrix)
    if n == 0: return True 
    if n == 1: return True 

    visited = [False]*n
    q = []
    
    start_node_bfs = -1
    for i in range(n): 
        if any(undirected_adj_matrix[i]): 
            start_node_bfs = i
            break
    if start_node_bfs == -1 : # Graph has no edges (all isolated nodes)
        return True # Or False if n > 1 meaning not connected. Conventionally, graph with >1 node and 0 edges is not connected.
                    # Let's assume for n>1, 0 edges is not connected in the typical sense.
                    # However, for Euler, we care about the component with edges.
                    # This function is a general connectivity check.
                    # If n > 1 and no edges, it will return True here, which might need adjustment based on context of use.
                    # For now, let it be. If it returns true for 0 edges n>1, then a check for num_nodes == count_visited_nodes might fail
                    # if start_node_bfs is -1. Let's ensure count_visited_nodes logic works for this.
        return True # If no edges, all nodes considered "visited" in their own trivial components.
                    # Or, more strictly, if n > 1 and no edges, it's not connected.
                    # If we must pick a start node, and there are no edges, the BFS won't explore.

    q.append(start_node_bfs)
    visited[start_node_bfs] = True
    count_visited_nodes = 0; head = 0
    
    while head < len(q):
        u = q[head]; head += 1; count_visited_nodes +=1
        for v_neighbor in range(n):
            if undirected_adj_matrix[u][v_neighbor] and not visited[v_neighbor]:
                visited[v_neighbor] = True; q.append(v_neighbor)
            
    # Check if all nodes that *should* be part of the main component (if edges exist) are visited.
    # If there are edges, we expect all N nodes to be visited if connected.
    # If there are no edges and N > 1, count_visited_nodes will be 0 if we don't start, or 1 if we pick an arbitrary start.
    # The `start_node_bfs == -1` check handles the no-edge case.
    # If edges exist, `start_node_bfs` is valid.
    num_nodes_with_any_connection_or_isolated = n # We want to see if all nodes are reachable.
    
    # If a graph has isolated nodes AND a connected component, is_connected should be false.
    # BFS from a node in the component will only visit that component.
    # So, count_visited_nodes should equal n ONLY if all nodes are in that one component.

    return count_visited_nodes == n 

def dfs(adj_matrix, node, visited, order_list=None):
    visited[node] = True
    if order_list is not None: order_list.append(node)
    for neighbor in range(len(adj_matrix)):
        if adj_matrix[node][neighbor] and not visited[neighbor]:
            dfs(adj_matrix, neighbor, visited, order_list)

def bfs(adj_matrix, start):
    n = len(adj_matrix)
    if n == 0: return []
    if not (0 <= start < n):
        if n > 0: start = 0 
        else: return [] 
    
    visited = [False]*n; queue = [start]; visited[start] = True; order = []; head = 0
    while head < len(queue):
        node = queue[head]; head += 1; order.append(node)
        for neighbor in range(n):
            if adj_matrix[node][neighbor] and not visited[neighbor]:
                visited[neighbor] = True; queue.append(neighbor)
    return order

def find_euler_path(undirected_adj_matrix): 
    n = len(undirected_adj_matrix)
    if n == 0: return None

    adj_list_hierholzer = [[] for _ in range(n)]
    num_edges_hierholzer = 0
    for i_node in range(n):
        for j_node in range(i_node, n): 
            if undirected_adj_matrix[i_node][j_node] == 1:
                adj_list_hierholzer[i_node].append(j_node)
                if i_node != j_node: 
                    adj_list_hierholzer[j_node].append(i_node)
                num_edges_hierholzer += 1 

    if n > 0 and num_edges_hierholzer == 0 : 
        return [0] if n == 1 else None # For multiple isolated nodes, no clear Euler path covering "all edges" (none exist)

    start_node_hierholzer = 0 
    degrees_hier = calculate_undirected_degrees(undirected_adj_matrix)
    
    found_start_node = False
    # Try to find an odd degree node if one exists (for an Euler path)
    for i_node in range(n):
        if degrees_hier[i_node] % 2 != 0:
            start_node_hierholzer = i_node
            found_start_node = True
            break
    if not found_start_node: 
        # If all degrees are even, pick any node with degree > 0 (for an Euler circuit)
        for i_node in range(n):
            if degrees_hier[i_node] > 0: 
                start_node_hierholzer = i_node
                found_start_node = True
                break
    
    if not found_start_node : 
        # This case means all degrees are 0.
        # If n=1, it's a single isolated node, num_edges_hierholzer was 0, handled above.
        # If n > 1 and all degrees 0, num_edges_hierholzer was 0, handled above.
        return None # Should have been caught by num_edges_hierholzer == 0

    path_hierholzer = []
    stack_hierholzer = [start_node_hierholzer]
    current_adj_list_hier = [list(neighbors) for neighbors in adj_list_hierholzer] 

    while stack_hierholzer:
        u_curr_hier = stack_hierholzer[-1]
        if current_adj_list_hier[u_curr_hier]: 
            v_next_hier = current_adj_list_hier[u_curr_hier].pop(0) 
            # Remove edge from v_next_hier's list as well for undirected graph
            if u_curr_hier != v_next_hier : # Avoid issues with self-loops if pop modifies same list
                if u_curr_hier in current_adj_list_hier[v_next_hier]: 
                    current_adj_list_hier[v_next_hier].remove(u_curr_hier)
            # else: # self-loop, already popped from u_curr_hier's list
            stack_hierholzer.append(v_next_hier) 
        else: 
            path_hierholzer.append(stack_hierholzer.pop())
            
    path_hierholzer.reverse() 

    # Ensure all edges were covered (Hierholzer should do this if graph component is connected and degrees are right)
    # The number of edges in the path should be equal to total edges in the graph component with edges.
    if num_edges_hierholzer > 0 and (len(path_hierholzer) -1) != num_edges_hierholzer:
        return None # Path found doesn't cover all edges

    return path_hierholzer

def has_hamilton_path(adj_matrix_hamilton):
    n = len(adj_matrix_hamilton)
    if n == 0: return False, None, None
    
    # Check for Hamiltonian Path
    for start_node_idx in range(n):
        path_list_ham = [start_node_idx]
        visited_nodes_ham = [False]*n
        visited_nodes_ham[start_node_idx] = True
        if hamilton_path_util(adj_matrix_hamilton, start_node_idx, visited_nodes_ham, path_list_ham, n):
            return True, "path", path_list_ham
            
    # Check for Hamiltonian Cycle
    for start_node_idx in range(n):
        path_list_ham = [start_node_idx] 
        visited_nodes_ham = [False]*n
        visited_nodes_ham[start_node_idx] = True 
        if hamilton_cycle_util(adj_matrix_hamilton, start_node_idx, start_node_idx, visited_nodes_ham, path_list_ham, n, 1):
            path_list_ham.append(start_node_idx) # Complete the cycle in the list
            return True, "cycle", path_list_ham
            
    return False, None, None

def hamilton_path_util(adj_matrix_h_util, v_curr_h, visited_nodes_h, path_list_h, n_total_h):
    if len(path_list_h) == n_total_h: return True 
    
    for i_neighbor_h in range(n_total_h):
        if adj_matrix_h_util[v_curr_h][i_neighbor_h] and not visited_nodes_h[i_neighbor_h]:
            visited_nodes_h[i_neighbor_h] = True
            path_list_h.append(i_neighbor_h)
            
            if hamilton_path_util(adj_matrix_h_util, i_neighbor_h, visited_nodes_h, path_list_h, n_total_h):
                return True
            
            # Backtrack
            visited_nodes_h[i_neighbor_h] = False
            path_list_h.pop()
    return False

def hamilton_cycle_util(adj_matrix_hc_util, start_node_hc, current_node_hc, visited_nodes_hc, path_list_hc, n_total_hc, count_visited_hc):
    if count_visited_hc == n_total_hc:
        # All nodes visited, check if there's an edge back to the start
        if adj_matrix_hc_util[current_node_hc][start_node_hc] == 1:
            return True
        else:
            return False

    for neighbor_idx_hc in range(n_total_hc):
        if adj_matrix_hc_util[current_node_hc][neighbor_idx_hc] == 1 and not visited_nodes_hc[neighbor_idx_hc]:
            visited_nodes_hc[neighbor_idx_hc] = True
            path_list_hc.append(neighbor_idx_hc)
            
            if hamilton_cycle_util(adj_matrix_hc_util, start_node_hc, neighbor_idx_hc, visited_nodes_hc, path_list_hc, n_total_hc, count_visited_hc + 1):
                return True
            
            # Backtrack
            visited_nodes_hc[neighbor_idx_hc] = False
            path_list_hc.pop()
    return False

def find_hamilton_path(undirected_adj_matrix_fhp): 
    has_path_bool_fhp, _, path_nodes_fhp = has_hamilton_path(undirected_adj_matrix_fhp)
    return path_nodes_fhp if has_path_bool_fhp else None

def prim_algorithm(undirected_weight_matrix_prim, start_node_prim_idx=0):
    n_nodes = len(undirected_weight_matrix_prim)
    if n_nodes == 0: return [[0]*n_nodes for _ in range(n_nodes)], 0

    if not (0 <= start_node_prim_idx < n_nodes):
        if n_nodes > 0: start_node_prim_idx = 0
        else: return [[0]*n_nodes for _ in range(n_nodes)], 0 

    adj_weights_prim = [[(undirected_weight_matrix_prim[i][j] if undirected_weight_matrix_prim[i][j] > 0 else float('inf')) 
                         for j in range(n_nodes)] for i in range(n_nodes)]
    
    key_values_prim = [float('inf')]*n_nodes 
    parent_nodes_prim = [-1]*n_nodes      
    key_values_prim[start_node_prim_idx] = 0  
    mst_set_nodes_prim = [False]*n_nodes    

    for _ in range(n_nodes): 
        min_key_val_prim = float('inf')
        u_selected_prim = -1 
        for v_node_idx_prim in range(n_nodes):
            if not mst_set_nodes_prim[v_node_idx_prim] and key_values_prim[v_node_idx_prim] < min_key_val_prim:
                min_key_val_prim = key_values_prim[v_node_idx_prim]
                u_selected_prim = v_node_idx_prim
        
        if u_selected_prim == -1: break # All reachable nodes processed

        mst_set_nodes_prim[u_selected_prim] = True 

        for v_node_idx_prim in range(n_nodes):
            if adj_weights_prim[u_selected_prim][v_node_idx_prim] != float('inf') and \
               not mst_set_nodes_prim[v_node_idx_prim] and \
               adj_weights_prim[u_selected_prim][v_node_idx_prim] < key_values_prim[v_node_idx_prim]:
                key_values_prim[v_node_idx_prim] = adj_weights_prim[u_selected_prim][v_node_idx_prim]
                parent_nodes_prim[v_node_idx_prim] = u_selected_prim
                
    mst_result_matrix_prim = [[0]*n_nodes for _ in range(n_nodes)]
    calculated_total_weight_prim = 0
    for i_node_prim in range(n_nodes):
        if parent_nodes_prim[i_node_prim] != -1: 
            p_node_prim = parent_nodes_prim[i_node_prim]
            # Weight should be from original undirected_weight_matrix_prim as adj_weights_prim has inf
            weight_of_edge_prim = undirected_weight_matrix_prim[i_node_prim][p_node_prim]
            
            if weight_of_edge_prim > 0: 
                mst_result_matrix_prim[i_node_prim][p_node_prim] = int(weight_of_edge_prim)
                mst_result_matrix_prim[p_node_prim][i_node_prim] = int(weight_of_edge_prim)
                calculated_total_weight_prim += weight_of_edge_prim
                
    return mst_result_matrix_prim, int(calculated_total_weight_prim)

def kruskal_algorithm(undirected_weight_matrix_kruskal):
    n_nodes = len(undirected_weight_matrix_kruskal)
    if n_nodes == 0: return [[0]*n_nodes for _ in range(n_nodes)], 0
    
    edge_list_kruskal = []
    for i_node_k in range(n_nodes):
        for j_node_k in range(i_node_k + 1, n_nodes): 
            if undirected_weight_matrix_kruskal[i_node_k][j_node_k] > 0:
                edge_list_kruskal.append((undirected_weight_matrix_kruskal[i_node_k][j_node_k], i_node_k, j_node_k))
    edge_list_kruskal.sort() 

    parent_uf_k = list(range(n_nodes)); rank_uf_k = [0]*n_nodes 
    def find_set_k(item_k): 
        if parent_uf_k[item_k] == item_k: return item_k
        parent_uf_k[item_k] = find_set_k(parent_uf_k[item_k]); return parent_uf_k[item_k]
    def unite_sets_k(item1_k, item2_k): 
        root1_k = find_set_k(item1_k); root2_k = find_set_k(item2_k)
        if root1_k != root2_k:
            if rank_uf_k[root1_k] < rank_uf_k[root2_k]: parent_uf_k[root1_k] = root2_k
            elif rank_uf_k[root1_k] > rank_uf_k[root2_k]: parent_uf_k[root2_k] = root1_k
            else: parent_uf_k[root2_k] = root1_k; rank_uf_k[root1_k] += 1
            return True
        return False 

    mst_result_matrix_kruskal = [[0]*n_nodes for _ in range(n_nodes)]
    total_mst_weight_kruskal = 0
    num_mst_edges_added_k = 0
        
    for weight_k, u_node_k, v_node_edge_k in edge_list_kruskal:
        if unite_sets_k(u_node_k, v_node_edge_k): 
            mst_result_matrix_kruskal[u_node_k][v_node_edge_k] = int(weight_k)
            mst_result_matrix_kruskal[v_node_edge_k][u_node_k] = int(weight_k)
            total_mst_weight_kruskal += weight_k
            num_mst_edges_added_k += 1
    return mst_result_matrix_kruskal, int(total_mst_weight_kruskal)

def dijkstra_algorithm(weight_matrix_dijkstra, start_node_index_d):
    n_nodes = len(weight_matrix_dijkstra)
    if n_nodes == 0: return [], []
    if not (0 <= start_node_index_d < n_nodes): 
        return [float('inf')] * n_nodes, [[] for _ in range(n_nodes)]

    dist_values_d = [float('inf')]*n_nodes
    prev_nodes_path_d = [-1]*n_nodes 
    dist_values_d[start_node_index_d] = 0
    priority_queue_d = [(0, start_node_index_d)] 

    while priority_queue_d:
        current_dist_d, u_node_d = heapq.heappop(priority_queue_d)
        if current_dist_d > dist_values_d[u_node_d]: continue 
        
        for v_node_neighbor_d in range(n_nodes):
            edge_weight_d = weight_matrix_dijkstra[u_node_d][v_node_neighbor_d]
            if edge_weight_d > 0: # Considers only positive weights, standard for Dijkstra algorith,
                if dist_values_d[u_node_d] + edge_weight_d < dist_values_d[v_node_neighbor_d]:
                    dist_values_d[v_node_neighbor_d] = dist_values_d[u_node_d] + edge_weight_d
                    prev_nodes_path_d[v_node_neighbor_d] = u_node_d
                    heapq.heappush(priority_queue_d, (dist_values_d[v_node_neighbor_d], v_node_neighbor_d))
    
    paths_reconstructed_d = []
    for i_target_node_d in range(n_nodes):
        path_single_d = []
        current_trace_node_d = i_target_node_d
        if dist_values_d[i_target_node_d] == float('inf'): 
            paths_reconstructed_d.append([])
            continue
            
        while current_trace_node_d != -1: 
            path_single_d.append(current_trace_node_d)
            current_trace_node_d = prev_nodes_path_d[current_trace_node_d]
        paths_reconstructed_d.append(path_single_d[::-1]) 
        
    return dist_values_d, paths_reconstructed_d

def draw_search_graph(adj_matrix_draw, search_type_name_draw, order_visited_draw, weight_matrix_draw=None, graph_type_label=""):
    common_node_attrs = {'penwidth': '1.0', 'fontname': 'Arial'}
    common_edge_attrs = {'penwidth': '0.8', 'fontname': 'Arial'}

    if graph_type_label == "(No Dirigido)":
        dot_graph_draw = Graph(name=f'{search_type_name_draw}_search_graph', engine='neato')
        dot_graph_draw.graph_attr.update({'splines': 'true', 'overlap': 'scale', 'sep': '0.6', 'K':'0.8'})
    else:
        dot_graph_draw = Digraph(name=f'{search_type_name_draw}_search_graph')
        dot_graph_draw.graph_attr.update({'splines': 'spline', 'ranksep': '0.75', 'nodesep': '0.5', 'concentrate':'true'})
    
    dot_graph_draw.node_attr.update(common_node_attrs)
    dot_graph_draw.edge_attr.update(common_edge_attrs)

    n_nodes = len(adj_matrix_draw); node_labels_char_draw = [chr(97 + i) for i in range(n_nodes)]
    for i, node_char_label_d in enumerate(node_labels_char_draw):
        dot_graph_draw.node(node_char_label_d, label=node_char_label_d, shape='ellipse')
    
    drawn_edges_undir_set = set() 
    for r_idx in range(n_nodes):
        start_j_loop = r_idx if graph_type_label == "(No Dirigido)" else 0
        for c_idx in range(start_j_loop, n_nodes):
            if adj_matrix_draw[r_idx][c_idx]:
                u_char, v_char = node_labels_char_draw[r_idx], node_labels_char_draw[c_idx]
                
                if graph_type_label == "(No Dirigido)" and r_idx != c_idx: 
                    edge_tuple = tuple(sorted((u_char, v_char)))
                    if edge_tuple in drawn_edges_undir_set: continue
                    drawn_edges_undir_set.add(edge_tuple)
                
                edge_attributes_draw = {}
                if weight_matrix_draw and weight_matrix_draw[r_idx][c_idx] > 0:
                    edge_attributes_draw['label'] = str(weight_matrix_draw[r_idx][c_idx])
                dot_graph_draw.edge(u_char, v_char, **edge_attributes_draw)

    if order_visited_draw: 
        for step_num_d, node_idx_visited_d in enumerate(order_visited_draw):
            if 0 <= node_idx_visited_d < n_nodes:
                node_char_current_d = node_labels_char_draw[node_idx_visited_d]
                dot_graph_draw.node(node_char_current_d, label=f"{node_char_current_d}\n({step_num_d+1})", style='filled', fillcolor='lightblue')
    
    search_path_display_str_d = " -> ".join([node_labels_char_draw[i] for i in order_visited_draw if 0 <= i < n_nodes]) if order_visited_draw else "N/A"
    title_str_d = f"Busqueda en {search_type_name_draw} {graph_type_label}\nOrden: {search_path_display_str_d}"
    dot_graph_draw.attr(label=title_str_d, labelloc='t', fontsize='14', fontname='Arial Bold')
    return dot_graph_draw

def draw_mst_graph(undirected_adj_matrix_mst_base, mst_adj_matrix_draw, total_mst_weight_draw, algorithm_id_name_mst, undirected_weight_matrix_mst_orig=None):
    dot_graph_mst = Graph(name=f'{algorithm_id_name_mst}_mst_graph', engine='neato')
    dot_graph_mst.graph_attr.update({'splines': 'true', 'overlap': 'scale', 'sep': '0.6', 'K': '0.8'})
    dot_graph_mst.node_attr.update({'penwidth': '1.0', 'fontname': 'Arial'})
    dot_graph_mst.edge_attr.update({'penwidth': '0.8', 'fontname': 'Arial'})

    n_nodes = len(undirected_adj_matrix_mst_base); node_labels_char_mst = [chr(97 + i) for i in range(n_nodes)]
    for node_char_label_m in node_labels_char_mst:
        dot_graph_mst.node(node_char_label_m, label=node_char_label_m, shape='ellipse')
    
    # Draw non-MST edges from original graph first (greyed out)
    if undirected_weight_matrix_mst_orig:
        for r_idx in range(n_nodes):
            for c_idx in range(r_idx + 1, n_nodes): # Undirected, so iterate once
                if undirected_weight_matrix_mst_orig[r_idx][c_idx] > 0 and \
                   not (mst_adj_matrix_draw[r_idx][c_idx] > 0): # Edge exists in original but not in MST
                    dot_graph_mst.edge(node_labels_char_mst[r_idx], node_labels_char_mst[c_idx],
                                       label=str(undirected_weight_matrix_mst_orig[r_idx][c_idx]), color='lightgray')
    
    # Draw MST edges (highlighted)
    for r_idx in range(n_nodes):
        for c_idx in range(r_idx + 1, n_nodes): # Undirected
            if mst_adj_matrix_draw[r_idx][c_idx] > 0:
                dot_graph_mst.edge(node_labels_char_mst[r_idx], node_labels_char_mst[c_idx],
                                   label=str(mst_adj_matrix_draw[r_idx][c_idx]), color='red', penwidth='2.5')
        # For self-loops if they were ever part of an MST (unlikely for standard MSTs, but if matrix contains)
        if mst_adj_matrix_draw[r_idx][r_idx] > 0: 
            dot_graph_mst.edge(node_labels_char_mst[r_idx], node_labels_char_mst[r_idx],
                                   label=str(mst_adj_matrix_draw[r_idx][r_idx]), color='red', penwidth='2.5')

    title_str_mst = f"Alg. de {algorithm_id_name_mst} (No Dirigido)\nArbol de Expansion Minima\nPeso total: {total_mst_weight_draw}"
    dot_graph_mst.attr(label=title_str_mst, labelloc='t', fontsize='14', fontname='Arial Bold')
    return dot_graph_mst

def draw_dijkstra_graph(undirected_adj_matrix_dk_base, dist_values_list_dk, paths_list_data_dk,
                        start_node_dijkstra_idx_dk, undirected_weight_matrix_dk_orig=None, # Corrected: added undirected_weight_matrix_dk_orig
                        end_node_dijkstra_idx_dk=None):
    dot_graph_dk = Graph(name='dijkstra_undirected_graph', engine='neato')
    dot_graph_dk.graph_attr.update({'splines': 'true', 'overlap': 'scale', 'sep': '0.6', 'K': '0.8'})
    dot_graph_dk.node_attr.update({'penwidth': '1.0', 'fontname': 'Arial'})
    dot_graph_dk.edge_attr.update({'penwidth': '0.8', 'fontname': 'Arial'})

    n_nodes = len(undirected_adj_matrix_dk_base)
    node_labels_char_dk = [chr(97 + i) for i in range(n_nodes)]

    if not (0 <= start_node_dijkstra_idx_dk < n_nodes):
        dot_graph_dk.attr(label="Error: Nodo de inicio de Dijkstra invalido", labelloc='t', fontsize='14', fontname='Arial Bold')
        return dot_graph_dk
    start_node_char_label_dk = chr(97 + start_node_dijkstra_idx_dk)

    for i, node_char_current_dk in enumerate(node_labels_char_dk):
        distance_val_dk = dist_values_list_dk[i] if i < len(dist_values_list_dk) else float('inf')
        label_dist_str_dk = str(distance_val_dk) if distance_val_dk != float('inf') else "inf"
        node_display_text_dk = f"{node_char_current_dk}\nDist: {label_dist_str_dk}"
        fill_color_node_dk = 'white' # Default
        if i == start_node_dijkstra_idx_dk: fill_color_node_dk = 'lightgreen'
        elif end_node_dijkstra_idx_dk is not None and i == end_node_dijkstra_idx_dk: fill_color_node_dk = 'lightcoral'
        dot_graph_dk.node(node_char_current_dk, label=node_display_text_dk, shape='ellipse', style='filled', fillcolor=fill_color_node_dk)

    edges_in_shortest_paths_to_display = set()
    paths_to_draw_highlight = []
    if paths_list_data_dk:
        if end_node_dijkstra_idx_dk is not None: 
            if 0 <= end_node_dijkstra_idx_dk < len(paths_list_data_dk) and paths_list_data_dk[end_node_dijkstra_idx_dk]:
                paths_to_draw_highlight.append(paths_list_data_dk[end_node_dijkstra_idx_dk])
        else: # Show all paths from start node
            for path in paths_list_data_dk:
                if path: paths_to_draw_highlight.append(path)
        
        for path_seg in paths_to_draw_highlight:
            if len(path_seg) > 1:
                for k_path in range(len(path_seg) - 1):
                    u_path, v_path = sorted((path_seg[k_path], path_seg[k_path+1])) # Canonical form for undirected edge
                    edges_in_shortest_paths_to_display.add((u_path,v_path))

    if undirected_weight_matrix_dk_orig: # Use the original symmetrized weights for drawing all edges
        for r_idx in range(n_nodes):
            for c_idx in range(r_idx, n_nodes): # Iterate for undirected graph to draw each edge once
                if undirected_weight_matrix_dk_orig[r_idx][c_idx] > 0: # If an edge exists with a weight
                    current_edge_tuple_sorted = tuple(sorted((r_idx, c_idx)))
                    
                    is_on_shortest_path = current_edge_tuple_sorted in edges_in_shortest_paths_to_display
                    label_w = str(undirected_weight_matrix_dk_orig[r_idx][c_idx])
                    color_edge = 'red' if is_on_shortest_path else 'lightgray'
                    penwidth_edge = '2.5' if is_on_shortest_path else dot_graph_dk.edge_attr.get('penwidth', '0.8')
                    
                    dot_graph_dk.edge(node_labels_char_dk[r_idx], node_labels_char_dk[c_idx],
                                      label=label_w, color=color_edge, penwidth=penwidth_edge)

    title_str_dk = f"Dijkstra (No Dirigido)\nCaminos mas cortos desde {start_node_char_label_dk}"
    if end_node_dijkstra_idx_dk is not None:
        if 0 <= end_node_dijkstra_idx_dk < n_nodes:
            end_node_char_label_dk = chr(97 + end_node_dijkstra_idx_dk)
            title_str_dk += f" (objetivo: {end_node_char_label_dk})"
        else: title_str_dk += " (objetivo: NODO FINAL INVALIDO)"
    dot_graph_dk.attr(label=title_str_dk, labelloc='t', fontsize='14', fontname='Arial Bold')
    return dot_graph_dk

# START OF MODIFIED FUNCTION
def draw_euler_graph(undirected_adj_matrix_eg, euler_path_nodes_list_eg, undirected_weight_matrix_eg=None):
    dot_graph_eg = Graph(name='euler_undirected_graph', engine='neato')
    # Added outputorder which might help with layering, though Graphviz usually handles attribute merging.
    dot_graph_eg.graph_attr.update({'splines': 'true', 'overlap': 'scale', 'sep': '0.6', 'K': '0.8', 'outputorder': 'edgesfirst'})
    dot_graph_eg.node_attr.update({'penwidth': '1.0', 'fontname': 'Arial'})
    dot_graph_eg.edge_attr.update({'penwidth': '0.8', 'fontname': 'Arial'})

    n_nodes = len(undirected_adj_matrix_eg)
    node_labels_char_eg = [chr(97 + i) for i in range(n_nodes)]

    is_significant_euler_path = False
    path_type_display_str_eg = "Camino"

    if euler_path_nodes_list_eg:
        # A path needs at least one edge (so >1 node in list).
        # A single isolated node is a trivial case handled if n_nodes == 1.
        if len(euler_path_nodes_list_eg) > 1:
            is_significant_euler_path = True
            if euler_path_nodes_list_eg[0] == euler_path_nodes_list_eg[-1]:
                path_type_display_str_eg = "Ciclo"
        elif n_nodes == 1 and len(euler_path_nodes_list_eg) == 1: # Trivial path for a single isolated node
            is_significant_euler_path = True # Considered "valid" for plotting its state
            # path_type_display_str_eg remains "Camino"

    # Node Styling: Define all nodes first.
    # Highlighting will be applied by re-defining nodes in the path later if it exists.
    for i, node_char_label_eg in enumerate(node_labels_char_eg):
        dot_graph_eg.node(node_char_label_eg, label=node_char_label_eg, shape='ellipse')

    # Edge Styling: Draw all base graph edges first in a neutral style cause we dont want to fail the subject.
    if undirected_adj_matrix_eg:
        for r_idx in range(n_nodes):
            for c_idx in range(r_idx, n_nodes): # Iterate to avoid duplicate edges for undirected graph
                if undirected_adj_matrix_eg[r_idx][c_idx]:
                    u_char_base, v_char_base = node_labels_char_eg[r_idx], node_labels_char_eg[c_idx]
                    edge_attrs_base_eg = {'color': 'lightgray', 'penwidth': '1.2'} # Slightly thicker base for visibility
                    if undirected_weight_matrix_eg and undirected_weight_matrix_eg[r_idx][c_idx] > 0:
                        edge_attrs_base_eg['label'] = str(undirected_weight_matrix_eg[r_idx][c_idx])
                    dot_graph_eg.edge(u_char_base, v_char_base, **edge_attrs_base_eg)
    
    title_str_eg = ""
    if is_significant_euler_path and euler_path_nodes_list_eg:
        title_str_eg = f"{path_type_display_str_eg} de Euler (No Dirigido)"
        path_sequence_str = ' -> '.join([node_labels_char_eg[node_idx] for node_idx in euler_path_nodes_list_eg if 0 <= node_idx < n_nodes])
        title_str_eg += f"\nSecuencia: {path_sequence_str}"

        # Highlight nodes involved in the Euler path
        nodes_in_path_indices = set(euler_path_nodes_list_eg) # Get unique nodes in path
        for node_idx in nodes_in_path_indices:
            if 0 <= node_idx < n_nodes:
                node_char = node_labels_char_eg[node_idx]
                fill_color_node = 'lightblue' # Default for intermediate path nodes

                # Specific colors for start and end nodes
                if node_idx == euler_path_nodes_list_eg[0]: # Start node
                    fill_color_node = '#90EE90' # LightGreen
                
                # If it's a path (not a cycle) and this is the end node
                if len(euler_path_nodes_list_eg) > 1 and \
                   node_idx == euler_path_nodes_list_eg[-1] and \
                   euler_path_nodes_list_eg[0] != euler_path_nodes_list_eg[-1]:
                     fill_color_node = '#FFA07A' # LightSalmon (for distinct end node and because its a nice source of food in minecraft)
                
                # For a cycle, start and end node are the same, so it will be LightGreen.
                dot_graph_eg.node(node_char, style='filled', fillcolor=fill_color_node)

        # Highlight edges in the Euler path with sequence numbers.
        # These edge definitions will typically override or merge with the base grey edges.
        if len(euler_path_nodes_list_eg) > 1: # Path has edges
            for i_edge_eg in range(len(euler_path_nodes_list_eg) - 1):
                u_node_idx_eg = euler_path_nodes_list_eg[i_edge_eg]
                v_node_idx_edge_eg = euler_path_nodes_list_eg[i_edge_eg+1]
                
                edge_attributes_highlight_eg = {
                    'color': '#3366FF',      # A more vibrant blue
                    'penwidth': '3.0',       # Thicker for emphasis
                    'label': f" {i_edge_eg+1} ", # Sequence number, padded for clarity
                    'fontcolor': '#000080',  # Dark blue for label text
                    'fontsize': '9',        # Slightly smaller font for label
                    'style': 'bold'          # Make path edges bold
                }
                # Note: If base edges have weight labels, this sequence label will overwrite it for path edges.
                # This is usually acceptable as the sequence is key for Euler paths.
                
                dot_graph_eg.edge(node_labels_char_eg[u_node_idx_eg], 
                                  node_labels_char_eg[v_node_idx_edge_eg], 
                                  **edge_attributes_highlight_eg)
    else: # No significant Euler path/cycle found
        if n_nodes == 0: # Graph is empty
            title_str_eg = "Grafo Vacio (No Dirigido)\nNo hay Camino/Ciclo de Euler"
        else: # Graph has nodes, but no Euler path/cycle meeting criteria
            title_str_eg = "Grafo (No Dirigido)\nNo se encontro Camino/Ciclo de Euler significativo"
            # If you wanted to add an explicit message on the graph canvas itself (can be tricky with 'neato' layout):
            # dot_graph_eg.node("msg_no_euler", label="No Euler Path/Cycle Found", shape="box", 
            #                   style="filled", fillcolor="lightyellow", fontcolor="red", fontsize="12")
            # For now, relying on the title is cleaner.

    dot_graph_eg.attr(label=title_str_eg, labelloc='t', fontsize='14', fontname='Arial Bold')
    return dot_graph_eg
# END OF MODIFIED FUNCTION :)

def draw_hamilton_graph(undirected_adj_matrix_hg, hamilton_path_nodes_list_hg, undirected_weight_matrix_hg=None):
    dot_graph_hg = Graph(name='hamilton_undirected_graph', engine='neato')
    dot_graph_hg.graph_attr.update({'splines': 'true', 'overlap': 'scale', 'sep': '0.6', 'K': '0.8'})
    dot_graph_hg.node_attr.update({'penwidth': '1.0', 'fontname': 'Arial'})
    dot_graph_hg.edge_attr.update({'penwidth': '0.8', 'fontname': 'Arial'})

    n_nodes = len(undirected_adj_matrix_hg); node_labels_char_hg = [chr(97 + i) for i in range(n_nodes)]
    hamilton_edges_hg = set()
    path_type_display_str_hg = "Camino" 
    is_cycle_hamilton = False

    if hamilton_path_nodes_list_hg and len(hamilton_path_nodes_list_hg) > 1:
        # Check for cycle: path length n+1, start=end, all n unique nodes visited before returning
        if hamilton_path_nodes_list_hg[0] == hamilton_path_nodes_list_hg[-1] and \
           len(set(hamilton_path_nodes_list_hg[:-1])) == n_nodes and \
           len(hamilton_path_nodes_list_hg) == n_nodes + 1:
            is_cycle_hamilton = True
            path_type_display_str_hg = "Ciclo"
        # Check for path: path length n, all n unique nodes visited
        elif len(set(hamilton_path_nodes_list_hg)) == n_nodes and \
             len(hamilton_path_nodes_list_hg) == n_nodes:
            path_type_display_str_hg = "Camino" # is_cycle_hamilton remains False
        # else: it's not a full Hamilton path/cycle by these strict length/uniqueness criteria

        for k_edge_hg in range(len(hamilton_path_nodes_list_hg) - 1):
            u_hg, v_hg = sorted((hamilton_path_nodes_list_hg[k_edge_hg], hamilton_path_nodes_list_hg[k_edge_hg+1]))
            hamilton_edges_hg.add((u_hg, v_hg))

    nodes_in_hamilton_path_ordered_unique = []
    if hamilton_path_nodes_list_hg:
        temp_path_for_node_order = hamilton_path_nodes_list_hg
        if is_cycle_hamilton: 
            temp_path_for_node_order = hamilton_path_nodes_list_hg[:-1] # Don't count last node for labeling order if cycle
        
        seen_for_labeling = set()
        for node_idx_label in temp_path_for_node_order:
            if node_idx_label not in seen_for_labeling:
                nodes_in_hamilton_path_ordered_unique.append(node_idx_label)
                seen_for_labeling.add(node_idx_label)
            
    for i_node_hg, node_char_label_hg in enumerate(node_labels_char_hg):
        label_node = node_char_label_hg
        style_attrs_node = {'shape': 'ellipse'}
        if i_node_hg in nodes_in_hamilton_path_ordered_unique:
            try:
                order_in_path = nodes_in_hamilton_path_ordered_unique.index(i_node_hg) + 1
                label_node = f"{node_char_label_hg}\n({order_in_path})"
                style_attrs_node['style'] = 'filled'; style_attrs_node['fillcolor'] = 'lightgreen'
            except ValueError: pass # Should not happen if logic is correct
        dot_graph_hg.node(node_char_label_hg, label=label_node, **style_attrs_node)
    
    if undirected_adj_matrix_hg:
        for r_idx in range(n_nodes):
            for c_idx in range(r_idx, n_nodes): # Undirected
                if undirected_adj_matrix_hg[r_idx][c_idx]:
                    current_edge_tuple_hg_sorted = tuple(sorted((r_idx, c_idx)))
                    edge_attrs_hg = {}
                    if undirected_weight_matrix_hg and undirected_weight_matrix_hg[r_idx][c_idx] > 0:
                        edge_attrs_hg['label'] = str(undirected_weight_matrix_hg[r_idx][c_idx])
                    
                    if current_edge_tuple_hg_sorted in hamilton_edges_hg:
                        edge_attrs_hg['color'] = 'darkgreen'; edge_attrs_hg['penwidth'] = '2.5'
                    else:
                        edge_attrs_hg['color'] = 'lightgray'
                    dot_graph_hg.edge(node_labels_char_hg[r_idx], node_labels_char_hg[c_idx], **edge_attrs_hg)

    title_str_hg = f"{path_type_display_str_hg} de Hamilton (No Dirigido)" if hamilton_path_nodes_list_hg and len(hamilton_path_nodes_list_hg) > 0 else "No hay Camino/Ciclo Hamilton (No Dirigido)"
    if hamilton_path_nodes_list_hg and len(hamilton_path_nodes_list_hg) > 0:
        title_str_hg += f"\nSecuencia: {' -> '.join([chr(97+n_idx_h) for n_idx_h in hamilton_path_nodes_list_hg])}"
    dot_graph_hg.attr(label=title_str_hg, labelloc='t', fontsize='14', fontname='Arial Bold')
    return dot_graph_hg

def show_results(
    original_directed_matrix, relation_type_str_res, directed_degrees_res,
    colors_res, cycle_nodes_res, euler_path_data_res, hamilton_path_data_res,
    original_directed_weight_matrix_res,
    bfs_order_data_res, dfs_order_data_res,
    prim_mst_result_res, prim_total_weight_val_res,
    kruskal_mst_result_res, kruskal_total_weight_val_res,
    dijkstra_dist_result_res, dijkstra_paths_result_res, dijkstra_start_node_idx_res,
    dijkstra_end_node_idx_res,
    prim_start_node_idx_res=None
    ):
    print("\n" + "="*50); print("RESULTADOS DEL ANALISIS".center(50)); print("="*50)
    n_nodes_show = len(original_directed_matrix)

    print("\nPROPIEDADES (Grafo Dirigido Original):")
    print(f"Reflexiva: {'Si' if is_reflexive(original_directed_matrix) else 'No'}")
    print(f"Simetrica (Dirigida): {'Si' if is_symmetric(original_directed_matrix) else 'No'}")
    print(f"Antisimetrica: {'Si' if is_antisymmetric(original_directed_matrix) else 'No'}")
    print(f"Transitiva: {'Si' if is_transitive(original_directed_matrix) else 'No'}")
    print(f"\nTipo de relacion (Dirigida): {relation_type_str_res}")

    print("\nGRADOS DE NODOS (Grafo Dirigido Original):")
    for i, (in_deg, out_deg) in enumerate(directed_degrees_res):
        if 0 <= i < n_nodes_show:
            total_deg = in_deg + out_deg
            print(f"Nodo {chr(97 + i)}: Entrada={in_deg}, Salida={out_deg}, Total={total_deg}")

    print("\nCOLORACION (Interpretacion No Dirigida):")
    if colors_res and any(c != -1 for c in colors_res):
        for i, color_val_node in enumerate(colors_res):
            if 0 <= i < n_nodes_show: print(f"Nodo {chr(97 + i)}: {'Color ' + str(color_val_node + 1) if color_val_node!=-1 else 'Sin color'}")
        print(f"Total de colores usados: {max(colors_res)+1 if any(c != -1 for c in colors_res) else 0}")
    else: print("No se aplico coloracion o grafo vacio.")

    print("\nCICLO OPTIMO (Interpretacion No Dirigida):") # This refers to the cycle found by find_cycle
    if cycle_nodes_res: print(" -> ".join([chr(97 + node_idx) for node_idx in cycle_nodes_res if 0 <= node_idx < n_nodes_show]))
    else: print("NO SE ENCONTRARON CICLOS (o grafo pequeno para ciclos > 2 aristas, o el tipo de ciclo buscado no existe :( ))")

    # MODIFIED PRINT BLOCK FOR EULER
    print("\nALGORITMO DE EULER (Interpretacion No Dirigida):")
    if euler_path_data_res and len(euler_path_data_res) > 1 : 
        path_type_str_euler = "Camino" 
        if euler_path_data_res[0] == euler_path_data_res[-1]: 
            if len(euler_path_data_res) > 1: # A cycle needs at least one edge. Path [a,a] from loop a-a has len 2.
                path_type_str_euler = "Ciclo"
        print(f"{path_type_str_euler} de Euler encontrado:\n{' -> '.join([chr(97 + node_idx) for node_idx in euler_path_data_res if 0 <= node_idx < n_nodes_show])}")
    elif euler_path_data_res and n_nodes_show == 1 and len(euler_path_data_res) == 1: 
        print(f"Camino de Euler (trivial):\n{chr(97+euler_path_data_res[0])}")
    else: 
        print("No existe camino/ciclo de Euler significativo.") # Removed "por lo que no se grafico"
    # END OF MODIFIED PRINT BLOCK BECAUSE IT NOW SHOWS THE GRAPH WITH THE MESSAGE

    print("\nALGORITMO DE HAMILTON (Interpretacion No Dirigida):")
    if hamilton_path_data_res and len(hamilton_path_data_res) > 0:
        is_cycle_output_ham = False
        if len(hamilton_path_data_res) == n_nodes_show + 1 and \
           hamilton_path_data_res[0] == hamilton_path_data_res[-1] and \
           len(set(hamilton_path_data_res[:-1])) == n_nodes_show: 
            is_cycle_output_ham = True
        elif len(hamilton_path_data_res) == n_nodes_show and \
             len(set(hamilton_path_data_res)) == n_nodes_show: 
            is_cycle_output_ham = False       
        path_type_str_ham = "Ciclo" if is_cycle_output_ham else "Camino"
        print(f"{path_type_str_ham} de Hamilton encontrado:\n{' -> '.join([chr(97 + node_idx) for node_idx in hamilton_path_data_res if 0 <= node_idx < n_nodes_show])}")
    else: print("No existe camino/ciclo de Hamilton.")

    print("\nBUSQUEDA EN ANCHURA (BFS: Breadth-First Search) (Interpretacion No Dirigida, desde 'a'):")
    if bfs_order_data_res: print(f"Orden de visita: {' -> '.join([chr(97 + node_idx) for node_idx in bfs_order_data_res if 0 <= node_idx < n_nodes_show])}")
    else: print("No aplicable (ej. grafo vacio).")

    print("\nBUSQUEDA EN PROFUNDIDAD (DFS: Depth-First Search) (Interpretacion No Dirigida, desde 'a'):")
    if dfs_order_data_res: print(f"Orden de visita: {' -> '.join([chr(97 + node_idx) for node_idx in dfs_order_data_res if 0 <= node_idx < n_nodes_show])}")
    else: print("No aplicable (ej. grafo vacio).")

    symmetrized_weights_exist = False
    if original_directed_weight_matrix_res:
        temp_sym_weights = symmetrize_weight_matrix(original_directed_matrix, original_directed_weight_matrix_res)
        if temp_sym_weights and any(any(w > 0 for w in row) for row in temp_sym_weights):
            symmetrized_weights_exist = True

    if symmetrized_weights_exist:
        print("\n--- Algoritmos con Pesos (Interpretacion No Dirigida) ---")
        if prim_mst_result_res is not None and prim_total_weight_val_res is not None:
            prim_start_info = ""
            actual_prim_start_char = 'N/A' 
            if prim_start_node_idx_res is not None and 0 <= prim_start_node_idx_res < n_nodes_show:
                actual_prim_start_char = chr(97 + prim_start_node_idx_res)
            elif n_nodes_show > 0 and prim_start_node_idx_res == 0 : 
                actual_prim_start_char = chr(97 + 0)
            prim_start_info = f" (iniciado desde nodo '{actual_prim_start_char}')" if actual_prim_start_char != 'N/A' else ""
            print(f"\nALGORITMO DE PRIM - MST{prim_start_info}:\nPeso total: {prim_total_weight_val_res}\nAristas seleccionadas:")
            edges_found_prim = False
            if n_nodes_show > 0 and prim_mst_result_res:
                for i in range(len(prim_mst_result_res)):
                    for j in range(i+1, len(prim_mst_result_res)): 
                        if prim_mst_result_res[i][j] > 0:
                            print(f"  {chr(97+i)} -- {chr(97+j)} : {prim_mst_result_res[i][j]}"); edges_found_prim=True
            if not edges_found_prim and n_nodes_show > 0: print("  No se formo arbol (o es un solo nodo, o grafo sin aristas con peso).")
            elif n_nodes_show == 0: print("  Grafo vacio.")
        else: print("\nALGORITMO DE PRIM - MST: No ejecutado o sin resultado (ej. grafo sin aristas o pesos validos).")

        if kruskal_mst_result_res is not None and kruskal_total_weight_val_res is not None:
            print(f"\nALGORITMO DE KRUSKAL - MST:\nPeso total: {kruskal_total_weight_val_res}\nAristas seleccionadas:")
            edges_found_kruskal = False
            if n_nodes_show > 0 and kruskal_mst_result_res:
                for i in range(len(kruskal_mst_result_res)):
                    for j in range(i+1, len(kruskal_mst_result_res)): 
                        if kruskal_mst_result_res[i][j] > 0:
                            print(f"  {chr(97+i)} -- {chr(97+j)} : {kruskal_mst_result_res[i][j]}"); edges_found_kruskal=True
            if not edges_found_kruskal and n_nodes_show > 0: print("  No se formo arbol (o es un solo nodo, o grafo sin aristas con peso).")
            elif n_nodes_show == 0: print("  Grafo vacio.")
        else: print("\nALGORITMO DE KRUSKAL - MST: No ejecutado o sin resultado (ej. grafo sin aristas o pesos validos).")

        if dijkstra_dist_result_res is not None and dijkstra_paths_result_res is not None and dijkstra_start_node_idx_res is not None:
            if not (0 <= dijkstra_start_node_idx_res < n_nodes_show):
                print("\nALGORITMO DE DIJKSTRA - Caminos Mas Cortos: Nodo de inicio invalido.")
            else:
                print("\nALGORITMO DE DIJKSTRA - Caminos Mas Cortos:")
                start_char = chr(97 + dijkstra_start_node_idx_res)
                if dijkstra_end_node_idx_res is not None: 
                    if not (0 <= dijkstra_end_node_idx_res < n_nodes_show):
                        print(f"  Desde el nodo {start_char} hasta el nodo objetivo: NODO FINAL INVALIDO.")
                    else:
                        end_char = chr(97 + dijkstra_end_node_idx_res)
                        print(f"  Desde el nodo {start_char} hasta el nodo {end_char}:")
                        target_node = dijkstra_end_node_idx_res
                        dist = dijkstra_dist_result_res[target_node] if target_node < len(dijkstra_dist_result_res) else float('inf')
                        path_list = dijkstra_paths_result_res[target_node] if target_node < len(dijkstra_paths_result_res) else []
                        path_str = " -> ".join([chr(97 + n) for n in path_list if 0<=n<n_nodes_show]) if path_list else "N/A"
                        dist_str = str(dist) if dist != float('inf') else 'No alcanzable'
                        path_final_str = path_str if dist != float('inf') and path_list else 'N/A'
                        print(f"    A {chr(97+target_node)}: Distancia = {dist_str}, Camino = {path_final_str}")
                else: 
                    print(f"  Desde el nodo {start_char}:")
                    for target_node in range(n_nodes_show):
                        if 0 <= target_node < len(dijkstra_dist_result_res): 
                            dist = dijkstra_dist_result_res[target_node]
                            path_list = dijkstra_paths_result_res[target_node] if target_node < len(dijkstra_paths_result_res) else []
                            path_str = " -> ".join([chr(97 + n) for n in path_list if 0<=n<n_nodes_show]) if path_list else "N/A"
                            dist_str = str(dist) if dist != float('inf') else 'No alcanzable'
                            path_final_str = path_str if dist != float('inf') and path_list else 'N/A'
                            print(f"    A {chr(97+target_node)}: Distancia = {dist_str}, Camino = {path_final_str}")
        else: print("\nALGORITMO DE DIJKSTRA: No ejecutado, nodo de inicio no valido, o sin resultados.")
    else: print("\n--- Algoritmos con Pesos OMITIDOS (no se ingresaron pesos o no son validos para grafo no dirigido) ---")
    print("\n" + "="*50)

def analyze_graph(original_directed_matrix, original_directed_weight_matrix=None,
                  dijkstra_start_node_input=None, dijkstra_end_node_input=None,
                  prim_start_node_input=None):
    n_nodes = len(original_directed_matrix)
    undirected_adj_matrix = symmetrize_adjacency_matrix(original_directed_matrix)
    undirected_weight_matrix = symmetrize_weight_matrix(original_directed_matrix, original_directed_weight_matrix)

    undirected_degrees = calculate_undirected_degrees(undirected_adj_matrix)
    relation_type_id_orig = "general"
    if is_reflexive(original_directed_matrix) and is_symmetric(original_directed_matrix) and is_transitive(original_directed_matrix): relation_type_id_orig = "equivalencia"
    elif is_reflexive(original_directed_matrix) and is_antisymmetric(original_directed_matrix) and is_transitive(original_directed_matrix): relation_type_id_orig = "orden parcial"
    directed_degrees_val = calculate_degrees(original_directed_matrix)

    cycle_val_undir = find_cycle(undirected_adj_matrix) 
    colors_val_undir = graph_coloring(original_directed_matrix) 

    has_euler_bool_undir, _ = has_euler_path(undirected_adj_matrix)
    euler_path_nodes_undir = find_euler_path(undirected_adj_matrix) if has_euler_bool_undir else None

    _, _, hamilton_path_nodes_undir = has_hamilton_path(undirected_adj_matrix)

    bfs_start_node_calc = 0 if n_nodes > 0 else -1 
    bfs_order_undir = bfs(undirected_adj_matrix, bfs_start_node_calc) if bfs_start_node_calc != -1 else None

    dfs_order_undir = []
    dfs_start_node_calc = 0 if n_nodes > 0 else -1 
    if dfs_start_node_calc != -1:
        visited_dfs_undir = [False]*n_nodes; dfs(undirected_adj_matrix, dfs_start_node_calc, visited_dfs_undir, dfs_order_undir)
    else: dfs_order_undir = None

    prim_mst_res, prim_weight_res = None, None
    kruskal_mst_res, kruskal_weight_res = None, None
    dijkstra_dist_res, dijkstra_paths_res, dijkstra_start_res = None, None, None
    
    actual_prim_start_node_for_algo = 0 
    if n_nodes > 0:
        if prim_start_node_input is not None and 0 <= prim_start_node_input < n_nodes:
            actual_prim_start_node_for_algo = prim_start_node_input
    
    if undirected_weight_matrix and any(any(w > 0 for w in row) for row in undirected_weight_matrix):
        if n_nodes > 0:
            prim_mst_res, prim_weight_res = prim_algorithm(undirected_weight_matrix, actual_prim_start_node_for_algo)
            kruskal_mst_res, kruskal_weight_res = kruskal_algorithm(undirected_weight_matrix)
            if dijkstra_start_node_input is not None and (0 <= dijkstra_start_node_input < n_nodes):
                dijkstra_start_res = dijkstra_start_node_input
                dijkstra_dist_res, dijkstra_paths_res = dijkstra_algorithm(undirected_weight_matrix, dijkstra_start_res)

    show_results(
        original_directed_matrix, relation_type_id_orig, directed_degrees_val,
        colors_val_undir, cycle_val_undir, euler_path_nodes_undir, hamilton_path_nodes_undir,
        original_directed_weight_matrix,
        bfs_order_undir, dfs_order_undir,
        prim_mst_res, prim_weight_res, kruskal_mst_res, kruskal_weight_res,
        dijkstra_dist_res, dijkstra_paths_res, dijkstra_start_res, dijkstra_end_node_input,
        actual_prim_start_node_for_algo if n_nodes > 0 else None 
    )

    images_to_plot_data = []

    basic_dir_graph = draw_basic_graph(original_directed_matrix, relation_type_id_orig, directed_degrees_val, original_directed_weight_matrix)
    try:
        basic_dir_graph.render('temp_basic_directed', format='png', cleanup=True)
        images_to_plot_data.append({'path': 'temp_basic_directed.png', 'title': "Grafo Original (Dirigido)"})
    except Exception as e: print(f"Error al renderizar grafo básico: {e}")


    if n_nodes > 0: 
        colored_undir_graph = draw_colored_graph(undirected_adj_matrix, colors_val_undir, undirected_degrees, cycle_val_undir, undirected_weight_matrix)
        try:
            colored_undir_graph.render('temp_colored_undirected', format='png', cleanup=True)
            images_to_plot_data.append({'path': 'temp_colored_undirected.png', 'title': "Coloreado (No Dirigido)"})
        except Exception as e: print(f"Error al renderizar coloreado: {e}")

        # --- MODIFIED CALL FOR THE EULER PLOT --- HOLA OSVALDO
        euler_undir_graph_obj = draw_euler_graph(undirected_adj_matrix, euler_path_nodes_undir, undirected_weight_matrix)
        try:
            euler_undir_graph_obj.render('temp_euler_undirected', format='png', cleanup=True)
            images_to_plot_data.append({'path': 'temp_euler_undirected.png', 'title': "Euler (No Dirigido)"})
        except Exception as e: print(f"Error al renderizar Euler: {e}")
        # --- END MODIFIED CALL --- ADIOS OSVALDO

        if hamilton_path_nodes_undir and len(hamilton_path_nodes_undir) > 0:
            hamilton_undir_graph = draw_hamilton_graph(undirected_adj_matrix, hamilton_path_nodes_undir, undirected_weight_matrix)
            try:
                hamilton_undir_graph.render('temp_hamilton_undirected', format='png', cleanup=True)
                images_to_plot_data.append({'path': 'temp_hamilton_undirected.png', 'title': "Hamilton (No Dirigido)"})
            except Exception as e: print(f"Error al renderizar Hamilton: {e}")

        if undirected_weight_matrix and any(any(w > 0 for w in row) for row in undirected_weight_matrix):
            if bfs_order_undir:
                bfs_g = draw_search_graph(undirected_adj_matrix, "Anchura", bfs_order_undir, undirected_weight_matrix, "(No Dirigido)")
                try:
                    bfs_g.render('temp_bfs_undir', format='png', cleanup=True)
                    images_to_plot_data.append({'path': 'temp_bfs_undir.png', 'title': "BFS (No Dirigido)"})
                except Exception as e: print(f"Error al renderizar BFS: {e}")

            if dfs_order_undir:
                dfs_g = draw_search_graph(undirected_adj_matrix, "Profundidad", dfs_order_undir, undirected_weight_matrix, "(No Dirigido)")
                try:
                    dfs_g.render('temp_dfs_undir', format='png', cleanup=True)
                    images_to_plot_data.append({'path': 'temp_dfs_undir.png', 'title': "DFS (No Dirigido)"})
                except Exception as e: print(f"Error al renderizar DFS: {e}")

            if prim_mst_res is not None:
                prim_g = draw_mst_graph(undirected_adj_matrix, prim_mst_res, prim_weight_res, "Prim", undirected_weight_matrix)
                try:
                    prim_g.render('temp_prim_undir', format='png', cleanup=True)
                    images_to_plot_data.append({'path': 'temp_prim_undir.png', 'title': "Prim (No Dirigido)"})
                except Exception as e: print(f"Error al renderizar Prim: {e}")

            if kruskal_mst_res is not None:
                kruskal_g = draw_mst_graph(undirected_adj_matrix, kruskal_mst_res, kruskal_weight_res, "Kruskal", undirected_weight_matrix)
                try:
                    kruskal_g.render('temp_kruskal_undir', format='png', cleanup=True)
                    images_to_plot_data.append({'path': 'temp_kruskal_undir.png', 'title': "Kruskal (No Dirigido)"})
                except Exception as e: print(f"Error al renderizar Kruskal: {e}")

            if dijkstra_dist_res is not None and dijkstra_start_res is not None:
                dijkstra_g = draw_dijkstra_graph(undirected_adj_matrix, dijkstra_dist_res, dijkstra_paths_res, 
                                                 dijkstra_start_res, undirected_weight_matrix, dijkstra_end_node_input) # Passed undirected_weight_matrix
                try:
                    dijkstra_g.render('temp_dijkstra_undir', format='png', cleanup=True)
                    title_dk_plot = f"Dijkstra (ND) desde '{chr(97+dijkstra_start_res)}'"
                    if dijkstra_end_node_input is not None and (0 <= dijkstra_end_node_input < n_nodes):
                        title_dk_plot += f" a '{chr(97+dijkstra_end_node_input)}'"
                    images_to_plot_data.append({'path': 'temp_dijkstra_undir.png', 'title': title_dk_plot})
                except Exception as e: print(f"Error al renderizar Dijkstra: {e}")

    if not images_to_plot_data:
        if n_nodes > 0 : print("\nNo se generaron imagenes adicionales para mostrar.")
        # If n_nodes is 0, the basic_dir_graph is the only one attempted.
        # It will be handled by the next condition if it's the only one.
        # If images_to_plot_data is empty AND n_nodes is 0, it means basic_dir_graph failed or wasn't added.
        # The goal here is just to prevent trying to plot an empty figure.
        return

    # Special handling if only the empty base graph was generated
    if n_nodes == 0 and len(images_to_plot_data) == 1 and images_to_plot_data[0]['title'] == "Grafo Original (Dirigido)":
        print("\nGrafo vacio. La imagen del grafo original (vacio) se intento generar.")
        # We might still want to show this single empty graph image if it rendered.
        # The plotting code below will handle a single image.
        pass # Let it proceed to plotting code
    elif n_nodes == 0 and not images_to_plot_data: # Basic graph also failed for empty graph
         print("\nGrafo vacio. No se generaron imagenes.")
         return


    num_actual_graphs = len(images_to_plot_data)
    cols = 3 
    rows = (num_actual_graphs + cols - 1) // cols 

    figsize_w = cols * 5.5 
    figsize_h = rows * 5.0  
    if num_actual_graphs == 1: figsize_w, figsize_h = 7, 6
    elif num_actual_graphs == 2: figsize_w, figsize_h = 11, 5.5 
    elif num_actual_graphs > 0 and rows == 1 : figsize_h = 6.0 

    fig = plt.figure(figsize=(figsize_w, figsize_h))

    for i, data in enumerate(images_to_plot_data):
        ax = fig.add_subplot(rows, cols, i + 1)
        try:
            img = Image.open(data['path'])
            ax.imshow(img)
            ax.set_title(data['title'], fontweight='bold', fontsize=10)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"Error:\n{data['path']}\nno encontrada", ha='center', va='center', color='red', fontsize=9)
            ax.set_title(data['title'] + " (Error Imagen)", fontweight='bold', color='red', fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error al cargar\n{data['path']}:\n{e}", ha='center', va='center', color='red', fontsize=9)
            ax.set_title(data['title'] + " (Error General)", fontweight='bold', color='red', fontsize=10)
        ax.axis('off')

    for j in range(num_actual_graphs, rows * cols): 
        ax = fig.add_subplot(rows, cols, j + 1)
        ax.axis('off')

    plt.tight_layout(pad=2.0, h_pad=3.5, w_pad=2.5) 
    if rows > 1 or (rows == 1 and cols > 1) :
        plt.subplots_adjust(top=0.92 if rows > 1 else 0.88) 
    elif rows == 1 and cols == 1: 
        plt.subplots_adjust(top=0.90)

    fig.suptitle("Analisis del grafo", fontsize=16, fontweight='bold', fontname='Arial')
    plt.show()

if __name__ == "__main__":
    def get_dimensions():
        while True:
            try:
                n_str = input("\nIngrese el numero de dimensiones (nxn, 0-10): ").strip() 
                if not n_str: print("Error: No ingreso un valor."); continue
                n = int(n_str)
                if 0 <= n <= 10: return n
                elif n > 10: print("Error: El tamano maximo permitido es 10 porfavor reingrese las dimensiones.")
                else: print("Error: El numero de dimensiones debe ser entre 0 y 10.")
            except ValueError: print("Error: Ingrese un numero valido.")

    def display_matrix(matrix):
        n = len(matrix)
        if n == 0:
            print("  (Matriz vacía)")
            return
        print("    " + " ".join([chr(97 + j) for j in range(n)])) # Column headers
        for i in range(n):
            print(f"{chr(97 + i)}:  " + " ".join(map(str, matrix[i]))) # Row header and content

    def display_weight_matrix(adj_matrix, weight_matrix_to_display):
        n = len(adj_matrix)
        if n == 0:
            print("  (Matriz de pesos vacia)")
            return
        print("    " + " ".join([chr(97 + j) for j in range(n)]))
        for i in range(n):
            row_display = []
            for j in range(n):
                if adj_matrix[i][j] == 1 and weight_matrix_to_display[i][j] > 0:
                    row_display.append(str(weight_matrix_to_display[i][j]))
                else:
                    row_display.append('0') # Show 0 if no edge or weight is 0
            print(f"{chr(97 + i)}:  " + " ".join(row_display))

    def input_matrix(n_dim_input):
        if n_dim_input == 0: return []
        while True:
            print(f"\nIngrese la matriz de adyacencia {n_dim_input}x{n_dim_input} fila por fila (0 o 1 separados por espacio):")
            matrix = []
            for i in range(n_dim_input):
                while True:
                    elements_str = input(f"Fila {chr(97 + i)}: ").strip()
                    elements = elements_str.split()
                    if len(elements) != n_dim_input:
                        print(f"  Error: Necesita {n_dim_input} valores. Ingreso {len(elements)}."); continue
                    try:
                        row_numbers = list(map(int, elements))
                        if any(x not in (0,1) for x in row_numbers):
                            print("  Error: Solo se permiten valores 0 y 1."); continue
                        matrix.append(row_numbers); break
                    except ValueError:
                        print("  Error: Ingrese solo numeros (0 o 1) separados por espacios.")

            print("\nMatriz de adyacencia ingresada:")
            display_matrix(matrix)
            confirm = input("¿Es correcta? (s/n): ").lower().strip()
            if confirm == 's': return matrix
            elif confirm == 'n': print("\nReingrese la matriz...")
            else: print("Opcion no valida. Asumiendo 'n' y reingresando...");

    def input_weight_matrix(n_dim_input, adjacency_matrix):
        if n_dim_input == 0: return []
        while True:
            print("\nIngrese los pesos para las aristas existentes (donde la matriz de adyacencia es 1).")
            print("Los pesos deben estar entre 1 y 10. Para no-aristas, el peso sera 0 por defecto.")
            print("Deje en blanco y presione Enter para asignar un peso aleatorio (1-10) a una arista.")
            print("\nMatriz de Adyacencia de referencia:")
            display_matrix(adjacency_matrix)

            weight_matrix = [[0]*n_dim_input for _ in range(n_dim_input)]
            for i in range(n_dim_input):
                for j in range(n_dim_input):
                    if adjacency_matrix[i][j] == 1: 
                        while True:
                            try:
                                prompt = f"  Peso para arista {chr(97+i)} -> {chr(97+j)} (1-10, Enter para aleatorio): "
                                weight_str = input(prompt).strip()
                                if not weight_str: 
                                    weight_matrix[i][j] = random.randint(1,10)
                                    print(f"    -> Peso aleatorio asignado: {weight_matrix[i][j]}")
                                    break
                                weight = int(weight_str)
                                if not (1 <= weight <= 10):
                                    print("    Error: Peso debe estar entre 1 y 10.")
                                    continue
                                weight_matrix[i][j] = weight
                                break
                            except ValueError: print("    Error: Ingrese un numero valido o deje en blanco.")

            print("\nMatriz de Pesos ingresada (dirigida):")
            display_weight_matrix(adjacency_matrix, weight_matrix)

            confirm = input("¿Es correcta la matriz de pesos? (s/n): ").lower().strip()
            if confirm == 's': return weight_matrix
            elif confirm == 'n': print("\nReingrese la matriz de pesos...")
            else: print("Opcion no válida. Asumiendo 'n' y reingresando...");

    def generate_random_weights(n_dim_input, adjacency_matrix):
        if n_dim_input == 0: return []
        weight_matrix = [[0]*n_dim_input for _ in range(n_dim_input)]
        for i in range(n_dim_input):
            for j in range(n_dim_input):
                if adjacency_matrix[i][j] == 1:
                    weight_matrix[i][j] = random.randint(1, 10)
        return weight_matrix

    def main():
        print("Universidad de Guanajuato - DICIS"); print("="*100)
        print("ANALIZADOR DE GRAFOS MEJORADO MICELIO HOLA OSVALDO...".center(100)); print("="*100)

        while True:
            print("\nMENU PRINCIPAL:\n1. Analizar nueva matriz\n2. Salir del programa")
            option = input("Seleccione una opcion (1-2): ").strip()

            if option == '1':
                n_dim = get_dimensions()
                matrix_adj_orig = input_matrix(n_dim)

                weights_orig = None
                dijkstra_start_node_val = None
                dijkstra_end_node_val = None
                prim_start_node_val = None

                if n_dim > 0: 
                    add_weights_choice = input("\n¿Desea agregar pesos al grafo? (s/n): ").lower().strip()
                    if add_weights_choice == 's':
                        print("\nOpciones para ingresar pesos:\n1. Ingresar manualmente cada peso\n2. Generar todos los pesos aleatoriamente (1-10)")
                        weight_opt = input("Seleccione opcion para pesos (1-2): ").strip()
                        if weight_opt == '1':
                            weights_orig = input_weight_matrix(n_dim, matrix_adj_orig)
                        elif weight_opt == '2':
                            weights_orig = generate_random_weights(n_dim, matrix_adj_orig)
                            print("\nPesos generados aleatoriamente (dirigidos):\n")
                            display_weight_matrix(matrix_adj_orig, weights_orig)
                        else: print("Opcion no valida para pesos, se continuara sin pesos especificos.")

                        if weights_orig: 
                            print("\n--- Configuracion para Algoritmos con Pesos ---")
                            print("\nConfiguracion para Prim:")
                            while True:
                                try:
                                    node_range_str = f"({chr(97)}-{chr(97+n_dim-1)})" if n_dim > 0 else ""
                                    default_char = chr(97) if n_dim > 0 else ""
                                    s_char_prim = input(f"  Nodo INICIO para Prim {node_range_str} (Enter para default '{default_char}'): ").lower().strip()
                                    if not s_char_prim and n_dim > 0: 
                                        prim_start_node_val = 0; break
                                    elif not s_char_prim and n_dim == 0: # No default if no nodes
                                        prim_start_node_val = None; break;
                                    if len(s_char_prim) == 1 and 'a'<=s_char_prim<chr(97+n_dim):
                                        prim_start_node_val=ord(s_char_prim)-ord('a'); break
                                    else: print("    Nodo invalido. Intente de nuevo.")
                                except Exception as e: print(f"Error al procesar entrada: {e}")

                            print("\nConfiguracion para Dijkstra:")
                            while True:
                                try:
                                    node_range_str = f"({chr(97)}-{chr(97+n_dim-1)})" if n_dim > 0 else ""
                                    s_char_dijkstra = input(f"  Nodo INICIO para Dijkstra {node_range_str}: ").lower().strip()
                                    if not s_char_dijkstra: print("    Debe ingresar un nodo de inicio."); continue
                                    if len(s_char_dijkstra) == 1 and 'a'<=s_char_dijkstra<chr(97+n_dim):
                                        dijkstra_start_node_val=ord(s_char_dijkstra)-ord('a'); break
                                    else: print("    Nodo invalido. Intente de nuevo.")
                                except Exception as e: print(f"Error al procesar entrada: {e}")

                            if dijkstra_start_node_val is not None: 
                                while True:
                                    try:
                                        node_range_str = f"({chr(97)}-{chr(97+n_dim-1)})" if n_dim > 0 else ""
                                        e_char_dijkstra = input(f"  Nodo FINAL para Dijkstra {node_range_str} (Enter para mostrar caminos a todos): ").lower().strip()
                                        if not e_char_dijkstra: dijkstra_end_node_val=None; break 
                                        if len(e_char_dijkstra) == 1 and 'a'<=e_char_dijkstra<chr(97+n_dim):
                                            dijkstra_end_node_val=ord(e_char_dijkstra)-ord('a'); break
                                        else: print("    Nodo invalido. Intente de nuevo.")
                                    except Exception as e: print(f"Error al procesar entrada: {e}")
                
                analyze_graph(matrix_adj_orig, weights_orig,
                              dijkstra_start_node_val, dijkstra_end_node_val,
                              prim_start_node_val)

                continue_choice = input("\n¿Desea analizar otra matriz? (s/n): ").lower().strip()
                if continue_choice != 's':
                    print("\nGracias por usar el Analizador de Grafos Papu Pro. ¡Hasta pronto amigo koku :D !")
                    break 

            elif option == '2':
                print("\nGracias por usar el Analizador de Grafos Papu Pro. ¡Hasta pronto amigo koku :D !")
                break 
            else:
                print("\nOpcion no valida. Por favor, elija 1 o 2.")
    main()