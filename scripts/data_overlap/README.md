# Data Overlap Script (no dependencies on HELM)

There are 2 scripts, one with and one without Apache Beam.

## Installation

```bash
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3 venv

# Activate the virtual environment.
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Running

```bash

This needs to be run from the data overlap directory; i.e. cd scripts/data_overlap if you are at the top level HELM folder

Usage:

python [compute_data_overlap_metrics.py OR run_data_overlap_beam.py] --input-data <input_data> --scenario-data <scenario_data> --output-stats <output_stats> --input-format <input_format>

For instance, you can call this with The Pile, e.g. have:
    input_data  = 00.jsonl (download https://pile.eleuther.ai/)
    scenario_data = (example included with repo, but can use HELM to generate)
    output_stats = arbitrary output file name, e.g. "output_stats"
    input_format = the_pile


There are additional optional args:
--normalization default 
--tags tag1 tag2
```

## Beam API

Model developers should implement an Apache Beam pipeline that creates a `PCollection[str]` of documents, and then pass it to `ComputeAndWriteOverlapStats()` with the appropriate arguments.

Note: Each record in the `PCollection[str]` should contain an _entire_ document, not a single line from a document.

```python
with beam.Pipeline() as pipeline:
    _ = (
        pipeline
        # The model developer should modify these lines to read from the actual training set.
        | "Read" >> beam.io.ReadFromText(input_data)
        | "ExtractTextFromDocument" >> beam.Map(extract_text_from_document)
        # Call the HELM Overlap Apache Beam API.
        | "ComputeAndWriteOverlapStats" >> ComputeAndWriteOverlapStats(
            scenario_data_path=scenario_data,
            n_values=n_values,
            normalization=normalization,
            tags=tags
        )
    )
```

## Notes

The beam script does not support outputting contaminated ngrams yet.


## Docker

To create and run docker image:

docker build  . -t overlap_script
docker run -e INPUT_PATH=<input_path> -e SCENARIO_DATA=<scenario_data> -e OUTPUT_STATS=<output_stats> -e INPUT_FORMAT=<input_format> --rm -it  overlap_script:latest 

example with values:
docker run -e INPUT_PATH="input.json" -e SCENARIO_DATA="scenario_data" -e OUTPUT_STATS="output_stats" -e INPUT_FORMAT="the_pile" --rm -it  overlap_script:latest 