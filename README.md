# Guide to build graphrag with local LLM
This repo is my settings for using the local LLM with graphrag

## Environment
I'm using Ollama (llama3) on Windows and LM Studio (nomic-text-embed) for text embeddings

Please don't use WSL because it will have issues connecting to the service on Windows (LM studio)

## Steps
First, activate the conda enviroment
```
conda activate rag
```

Clone this project then cd the directory
```
git clone <this repo link>
cd <folder name>
```

Then pull the code of graphrag and install the package 
```
git clone https://github.com/microsoft/graphrag.git
pip install -e ./graphrag
```

You can skip this step if you used this repo, but this is for initializing the graphrag folder
```
python -m graphrag.index --init --root .
```

Create your `.env` file
```
cp .env.example .env
```

Move your input text to `./input/`

Double check the parameters in `.env` and `settings.yaml`, make sure in `setting.yaml`, 
it should be "community_reports" instead of "community_report"

Then tune the prompts (this is important, this will generate a much better result)
```
python -m graphrag.prompt_tune --root . --no-entity-types
```

Then you can start the indexing
```
python -m graphrag.index --root .
```

You can check the logs in `./output/<timestamp>/reports/indexing-engine.log` for errors

Test a global query
```
python -m graphrag.query \
--root . \
--method global \
"What are the top themes in this story?"
```

## Using the UI

First, make sure requirements are installed
```
pip install -r requirements.txt
```

Then run the app using 
```
gradio app.py
```

To use the app, visit http://127.0.0.1:7860/

Make sure you select the valid output folder before you query

Note that "/generate" will disregard the query type and generate questions with a local search