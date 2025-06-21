from tqdm import tqdm
import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='name of the file to be filtered')
    parser.add_argument('--output_file', type=str, help='name of the output file')

    args = parser.parse_args()


    exclude_sample = 0
    total_sample = 0
    with open(args.input, "r") as rf, open(args.output, "w") as wf:
        for line in tqdm(rf):
            data = json.loads(line)
            fid = str(data['id']) + "-" + data['func_name'].replace("_", "-")

            type_stripped_vars = data["type_stripped_vars"]
            count = 0
            for key, value in type_stripped_vars.items():
                if value == "dwarf":
                    pass
                else:
                    count += 1
            total_count = len(type_stripped_vars)
            if count / total_count < 1:
                json.dump(data, wf)
                wf.write("\n")
            else:
                exclude_sample += 1

            total_sample += 1
        print( "被排除的样本占比:", exclude_sample / total_sample)

