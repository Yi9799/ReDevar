import re

def process_line(line, last_assignment, never_assignment, instance_counter, data_origin):

    pattern = r"@@(var_\d+)@@([\w]+)@@"
    tokens = re.findall(pattern, line)
    if not tokens:
        return instance_counter

    assignment_pattern = r"(?<![=!<>])(?:\+|-|%|/|\*|\||\^|&|>>|<<)?=(?!=)"
    increment_pattern = r"\+\+|--"

    increment_match = re.search(increment_pattern, line)
    increment_pos = increment_match.start() if increment_match else float('inf')

    assignment_match = re.search(assignment_pattern, line)
    assignment_pos = assignment_match.start() if assignment_match else float('inf')

    if increment_pos < assignment_pos:
        if(len(tokens))==1:

            left_var = tokens[0][1]
            current_instance_id = instance_counter
            instance_counter += 1
            left_instance_id = last_assignment.get(left_var, -1)
            if left_instance_id == -1:

                left_instance_id = never_assignment.get(left_var, -1)
            data_origin.append(left_instance_id)
            last_assignment[left_var] = current_instance_id
        else:

            left_var = tokens[0][1]
            current_instance_id = instance_counter
            instance_counter += 1
            left_instance_id=last_assignment.get(left_var, -1)
            if left_instance_id == -1:

                left_instance_id = never_assignment.get(left_var, -1)
            data_origin.append(left_instance_id)

            last_assignment[left_var] = current_instance_id

            for i in range(1, len(tokens)):
                right_var = tokens[i][1]
                current_instance_id = instance_counter
                instance_counter += 1

                increment_match = re.search(
                    rf"(\+\+|--)\s*@@{tokens[i][0]}@@{right_var}@@|@@{tokens[i][0]}@@{right_var}@@\s*(\+\+|--)", line)

                if increment_match:
                    origin_id=last_assignment.get(right_var, -1)
                    if origin_id == -1:
                        origin_id = never_assignment.get(right_var, -1)
                    data_origin.append(origin_id)
                    last_assignment[right_var] = current_instance_id
                else:
                    origin_id = last_assignment.get(right_var, -1)
                    if origin_id == -1:

                        origin_id = never_assignment.get(right_var, -1)
                    data_origin.append(origin_id)

    elif assignment_pos < increment_pos:
        left_var = tokens[0][1]
        current_instance_id = instance_counter
        instance_counter += 1
        left_instance_id = current_instance_id

        right_vars = tokens[1:]
        if right_vars:

            data_origin.append(current_instance_id + 1)

            for i in range(1, len(tokens)):
                right_var = tokens[i][1]
                current_instance_id = instance_counter
                instance_counter += 1

                increment_match = re.search(
                    rf"(\+\+|--)\s*@@{tokens[i][0]}@@{right_var}@@|@@{tokens[i][0]}@@{right_var}@@\s*(\+\+|--)", line)

                if increment_match:

                    origin_id=last_assignment.get(right_var, -1)
                    if origin_id == -1:
                        origin_id = never_assignment.get(right_var, -1)
                    data_origin.append(origin_id)
                    last_assignment[right_var] = current_instance_id
                else:
                    origin_id = last_assignment.get(right_var, -1)
                    if origin_id == -1:
                        origin_id = never_assignment.get(right_var, -1)
                    data_origin.append(origin_id)
        else:
            data_origin.append(-1)

        last_assignment[left_var] = left_instance_id
        never_assignment.pop(left_var, None)

    else:
        for var_id, var_name in tokens:
            current_instance_id = instance_counter
            instance_counter += 1

            origin_id = last_assignment.get(var_name, -1)
            if origin_id == -1:
                origin_id = never_assignment.get(var_name, -1)
                if origin_id == -1:
                    never_assignment[var_name] = current_instance_id

            data_origin.append(origin_id)

    return instance_counter



def analyze_data_origin(code):

    last_assignment = {}
    data_origin = []
    instance_counter = 0
    never_assignment = {}

    lines = code.split("\n")

    for line in lines:
        sub_lines=line.split(";")
        if ";" not in line or sub_lines[1]==' ':
            instance_counter = process_line(line, last_assignment, never_assignment, instance_counter, data_origin)
        else:
            for line in sub_lines:
                instance_counter = process_line(line, last_assignment, never_assignment, instance_counter, data_origin)

    return data_origin


