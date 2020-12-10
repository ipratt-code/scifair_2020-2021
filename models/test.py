import yaml

best = {"a": 1, "b": 2, "c": 3}

with open(r"/home/ianpratt/github/scifair_2020-2021/models/usa.model-conf.yml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    mod_data = yaml.load(file, Loader=yaml.FullLoader)
    print(mod_data["flexible_params"])
    file.close()


with open(
    "/home/ianpratt/github/scifair_2020-2021/models/usa.model-conf.yml", "w"
) as file:
    print(mod_data["flexible_params"])
    for param in best:
        mod_data["flexible_params"][param] = float(best[param])
    yaml.dump(mod_data, file)

# return yaml.dump(mod_data)
