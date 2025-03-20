from pathlib import Path
import random
import yaml
from biotrainer.utilities import read_FASTA

def load_config_from_yaml(config_path: Path) -> dict:
    try:
        with open(config_path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
        if not isinstance(config_dict, dict):
            raise ValueError(f"YAML file {config_path} did not parse into a dictionary")
        return config_dict
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    except yaml.YAMlLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")


SUPPORTED_MODELS = ["gaussian_process"]


# consider error handling
def botrain(config_path: str):
    print(f"in botrain: {config_path}")
    config_dict = load_config_from_yaml(config_path)
    # print(config_dict)
    results = []
    all_seqs = {seq.id: str(seq.seq) for seq in read_FASTA(config_dict["sequence_file"])}
    for id, seq in all_seqs.items():
        results.append({"id": id, "sequence": seq, "score": float(f"{random.random():.6f}")})
    results.sort(key=lambda x: x["score"], reverse = True)
    # print(results)

    results_yaml = yaml.dump(results)
    out_file = Path(config_dict["output_dir"])/"out.yml"
    with out_file.open('w+') as config_file:
        config_file.write(results_yaml)
