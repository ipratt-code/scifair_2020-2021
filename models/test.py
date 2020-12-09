import yaml

with open(r"/home/ianpratt/github/scifair_2020-2021/models/usa.model-conf.yml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

for i in data["train"]["fixed_parameters"]:
    print(data["train"]["fixed_parameters"][i])
