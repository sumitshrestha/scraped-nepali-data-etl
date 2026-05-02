# Reddit ETL Script

This project provides a simple Python script to extract and transform Reddit post and comment data from JSON files into a normalized format.

## Features
- Extracts post and comment information from Reddit JSON data
- Handles both individual post files and listings
- Outputs a single `extracted_posts.json` file with normalized data
- Parameterized input directory via `.env` file
- Logging for progress and error tracking

## Requirements
- Python 3.7+
- See `requirements.txt` for dependencies

## Setup
1. Clone this repository or copy the script files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root:
   ```env
   SCRAPED_DIR=your_reddit_data_folder
   ```
   Replace `your_reddit_data_folder` with the path to your folder containing Reddit JSON files.

## Usage
Run the script from the project directory:

```bash
python reddit-etl.py
```

- The script will recursively process all `.json` files in the directory specified by `SCRAPED_DIR`.
- Output will be saved to `extracted_posts.json` in the project root.
- Progress and errors are logged to the console.

## Output Format
The output file `extracted_posts.json` is a list of posts, each with the following structure:

```json
{
  "id": "post_id",
  "title": "Post Title",
  "content": "Post body or selftext",
  "author": "AuthorName",
  "created_utc": 1234567890,
  "type": "post type",
  "comments": [
    {
      "id": "comment_id",
      "author": "CommentAuthor",
      "body": "Comment text",
      "created_utc": 1234567890
    }
  ]
}
```

## Notes
- Only standard Python libraries and `python-dotenv` are required.
- The `.env` and output files are excluded from version control via `.gitignore`.

## License
MIT License

## Run AI Enrichment Worker As A Docker Background Service

1. Copy `.env.sample` to `.env` and set worker values.
2. If MongoDB or Ollama is running on your host machine, do not keep `localhost` inside container settings.
3. Set these in `.env` when using host services from Docker:
  - `MONGO_URI=mongodb://host.docker.internal:27017/?replicaSet=rs0`
  - `OLLAMA_ENDPOINT=http://host.docker.internal:11434`

Build and run in detached mode:

```bash
docker compose -f docker-compose.ai-enrichment.yml up -d --build
```

View logs:

```bash
docker compose -f docker-compose.ai-enrichment.yml logs -f
```

Stop service:

```bash
docker compose -f docker-compose.ai-enrichment.yml down
```

Notes:
- Service name: `ai-enrichment-worker`
- Restart policy: `unless-stopped`
- File logs are persisted to the host folder `./logs`.
