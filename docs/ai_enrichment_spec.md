**AI Enrichment Worker**

Technical Specification

*Nepali Corpus --- Devanagari Transliteration & Profanity Scoring Pipeline*

**1. Architectural Overview**

The AI enrichment worker (ai_enrichment_worker.py) is a standalone, decoupled script that runs independently of the main ETL pipeline. It continuously polls MongoDB for corpus documents that have not yet received a Devanagari transliteration, processes them via a locally-hosted LLM (Ollama), and writes results back to MongoDB.

This decoupled design means the enrichment job can be paused, resumed, and scaled independently of ingestion. It has no side effects on the ETL pipeline and makes no assumptions about ingestion order or completeness.

The worker\'s responsibilities are, in order:

- Fetch unprocessed documents from MongoDB ordered by priority

- Prepare clean, masked input text for the LLM

- Route documents to single or batch LLM calls based on text length

- Post-process and validate LLM responses

- Rehydrate English placeholders in Python

- Bulk write results back to MongoDB

**2. Configuration & Environment Variables**

All configuration is supplied via environment variables (loaded from .env). This allows switching between local and cloud LLM providers without code changes.

|                             |                                                                                    |
|-----------------------------|------------------------------------------------------------------------------------|
| **Variable**                | **Description / Default**                                                          |
| LLM_PROVIDER                | LLM backend to use. Default: ollama                                                |
| OLLAMA_ENDPOINT             | Base URL of the Ollama instance, e.g. http://192.168.x.x:11434                     |
| LLM_MODEL                   | Model tag to use, e.g. gemma3:27b                                                  |
| SHORT_TEXT_CHAR_THRESHOLD   | Character length below which a prepared text is treated as short. Default: 40      |
| SHORT_TEXT_BATCH_SIZE       | Max number of short texts bundled into one batch prompt. Default: 20               |
| LLM_REQUEST_TIMEOUT_SECONDS | HTTP timeout for Ollama requests in seconds. Default: 120                          |
| MAX_RETRY_COUNT             | Max failures before a document is permanently skipped. Default: 3                  |
| WORDFREQ_ENGLISH_TOPN       | Top-N frequency-ranked English words used for token classification. Default: 40000 |
| MONGO_URI                   | MongoDB connection string (existing)                                               |
| MONGO_DB                    | Target database name (existing)                                                    |
| MONGO_COLLECTION            | Target collection name (existing)                                                  |

**Note:** *Character count is used as the routing threshold rather than token count because it is a more reliable proxy for LLM context consumption without requiring a tokenizer call per document.*

**3. Schema Changes**

**3.1 ai_slots --- New Fields**

Two new fields are added to the ai_slots object on every corpus document:

|                          |                                                                                                                                                                                           |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Field**                | **Description**                                                                                                                                                                           |
| ai_slots.profanity_score | float \| null --- LLM-assigned profanity score, 0.0 (clean) to 1.0 (highly profane). Null until enriched.                                                                                 |
| ai_slots.retry_count     | int --- Number of failed LLM attempts for this document. Default 0. Written by merge_etl at insert time via \$setOnInsert. Documents at or above MAX_RETRY_COUNT are permanently skipped. |

The merge_etl.py \$setOnInsert block must be updated to include:

> \"ai_slots.retry_count\": 0

**3.2 linguistic_profile.english_token_indices --- Existing Field**

This field already exists at the top level of every corpus document, populated by merge_etl.py during the reduce phase as a union of English token indices across all variants of the same cleaned_text. The enrichment worker reads from this field directly.

Important behaviours inherited from the ETL:

- When a document has zero English tokens, merge_etl omits the field entirely rather than writing an empty list. The enrichment worker treats a missing field and an empty list identically --- no masking needed, send the full prepared text to the LLM.

- Indices are positional --- they refer to whitespace-split token positions in cleaned_text. Token 0 is the first word, token 1 the second, and so on. The enrichment worker must use the same whitespace split to reconstruct positions.

- The variant-level english_token_indices fields inside each variant object are ETL artifacts only. The enrichment worker never reads from them.

**4. English Token Detection --- wordfreq**

The existing annotate_tokens() function in merge_etl.py and the \_COMMON_ENGLISH_WORDS frozenset in lang_filter.py use a small hardcoded word list for English token detection. The enrichment worker uses wordfreq instead.

> from wordfreq import top_n_list
>
> ENGLISH_WORDS = frozenset(top_n_list(\'en\', WORDFREQ_ENGLISH_TOPN))

wordfreq provides frequency-ranked vocabulary coverage. At a threshold of 40,000 words, common everyday English is well covered while rare English words that collide with Romanized Nepali tokens (e.g. dal, ko, la, ho) are excluded. The WORDFREQ_ENGLISH_TOPN config variable is the control knob for this trade-off.

**Follow-up recommendation:** *Replacing \_COMMON_ENGLISH_WORDS with wordfreq in lang_filter.py and merge_etl.py would improve English detection across the whole pipeline. This is out of scope for the enrichment worker but is worth scheduling as a follow-up ETL task.*

**Known limitation:** *The ETL\'s annotate_tokens() uses only \_COMMON_ENGLISH_WORDS, not wordfreq or Lingua. Less common English words and proper nouns embedded in Nepali text will not be present in linguistic_profile.english_token_indices and will reach the LLM unmasked. This is an accepted limitation inherited from ETL time. The enrichment worker does not attempt to supplement or repair these indices at runtime.*

**5. Processing Pipeline**

**5.1 Fetch Query**

Documents are fetched from MongoDB ordered by metadata.total_global_occurrences descending --- highest frequency terms first. This maximises coverage value per LLM call, since a transliteration of a term that appears 500 times is more valuable than one that appears once.

Fetch query:

> {
>
> \"ai_slots.devanagari_translation\": None,
>
> \"ai_slots.retry_count\": {\"\$lt\": MAX_RETRY_COUNT}
>
> }

At worker startup, log a count of documents currently at or above MAX_RETRY_COUNT so poisoned documents are visible without querying manually.

**5.2 Stage 1 --- Short-Circuit Checks**

Before any LLM call, evaluate two conditions that allow the document to be written immediately without involving the LLM:

- If linguistic_profile.has_devanagari is true: the text already contains Devanagari script. Write cleaned_text as-is to ai_slots.devanagari_translation and set ai_slots.model_version to \"passthrough-has-devanagari\". Skip all remaining stages for this document.

- If every whitespace-split token in cleaned_text is present in ENGLISH_WORDS: there is nothing to transliterate. Write cleaned_text as-is to ai_slots.devanagari_translation and set ai_slots.model_version to \"passthrough-all-english\". Skip all remaining stages for this document.

**5.3 Stage 2 --- Input Preparation**

The ETL pipeline (lang_filter.py clean_text()) has already removed Discord mentions, custom emoji, URLs, shortcode emoji, unicode emoji, decorative separators, and markdown symbols. Do not re-clean. cleaned_text is already clean when it arrives from MongoDB.

Preparation steps:

- Tokenize cleaned_text by whitespace

- Apply a lightweight safety-net strip per token: retain letters, digits, and semantically meaningful punctuation (. , ? ! \'). Strip anything else as a safety net only --- not as a primary cleaning pass. These punctuation characters carry emotional or semantic meaning and should reach the LLM intact.

- Read linguistic_profile.english_token_indices from the document. Treat a missing field and an empty list identically --- no masking needed.

- For each index in english_token_indices, replace that token in the prepared token list with a positional placeholder: \[ENG_0\], \[ENG_1\], \[ENG_2\], etc. The numbering is sequential in the order placeholders are introduced, not necessarily matching the original index value.

- Store a local rehydration map --- a dict mapping placeholder number to the original English word: {0: \"day\", 1: \"feel\", 2: \"so\"}.

- Join the prepared token list back into a string. This is the prepared text sent to the LLM.

**Important:** *English word preservation is entirely a Python responsibility handled at rehydration (Stage 5). The LLM is never trusted to preserve English words on its own --- it only needs to leave the \[ENG_N\] placeholders intact.*

**5.4 Stage 3 --- Routing**

Measure the character length of the prepared text after masking and safety-net stripping:

- If length \<= SHORT_TEXT_CHAR_THRESHOLD: add the document to the short text buffer

- If length \> SHORT_TEXT_CHAR_THRESHOLD: add the document to the single-item queue for immediate processing

Flush the short text buffer to the LLM when it reaches SHORT_TEXT_BATCH_SIZE items. Also flush any remaining buffer when the cursor exhausts --- do not discard leftover short texts at end of run.

**5.5 Stage 4 --- LLM Call**

Use Ollama\'s /api/generate endpoint with \"format\": \"json\" in the request payload. This constrains the model\'s output tokenizer to valid JSON natively, significantly reducing malformed responses. Keep the markdown-strip fallback in the parsing step regardless as a safety net.

Use \"temperature\": 0.1 in all requests for deterministic, consistent output.

See Section 6 for prompt templates.

**5.6 Stage 5 --- Post-Processing & Validation**

After the LLM responds:

- Strip any markdown code fences (\`\`\`json \... \`\`\`) as a safety net

- Parse the response string as JSON. On json.JSONDecodeError go to error handling (Section 7).

- For each result entry, validate that devanagari is a non-empty string and profanity_score is a float in range \[0.0, 1.0\]. On validation failure go to error handling.

- Validate that every \[ENG_N\] placeholder present in the prepared text also appears in the LLM\'s devanagari output. Missing placeholders are a validation failure.

- Rehydrate: replace each \[ENG_N\] placeholder in the LLM output with the original English word from the rehydration map. The result is the final Devanagari string stored in MongoDB.

**5.7 Stage 6 --- Bulk Write**

Collect completed results and bulk write to MongoDB using UpdateOne with ordered=False. Flush bulk write every 200 documents or at end of cursor, whichever comes first.

Each write operation:

> UpdateOne(
>
> {\"\_id\": doc_id},
>
> {\"\$set\": {
>
> \"ai_slots.devanagari_translation\": rehydrated_text,
>
> \"ai_slots.profanity_score\": profanity_score,
>
> \"ai_slots.model_version\": f\"ollama/{LLM_MODEL}\",
>
> \"ai_slots.translated_at\": time.time()
>
> }}
>
> )

Passthrough documents (from Stage 1 short-circuits) are also written via the same bulk write path, with profanity_score left null and translated_at set.

**6. Prompt Engineering**

**6.1 System Prompt (Both Single and Batch Calls)**

The same system prompt is used for all LLM calls:

> *You are an expert linguist in Nepali. Your task is script conversion, not translation --- convert Romanized Nepali tokens into Devanagari script. The text may be a mix of Romanized Nepali and placeholders in the format \[ENG_N\]. Leave all placeholders exactly as they appear including the brackets and number --- do not translate, alter, or remove them. The input may contain punctuation such as . , ? ! \' --- preserve these in their relative positions in your output. Evaluate profanity in the Nepali cultural and linguistic context. Common Nepali slang and colloquialisms must be scored by Nepali standards, not English-centric ones.*

**6.2 Single Item Prompt (Long Text)**

Used when the prepared text exceeds SHORT_TEXT_CHAR_THRESHOLD characters. One document per LLM call.

> Convert the following Romanized Nepali text to Devanagari script. Preserve all \[ENG_N\] placeholders exactly. Also evaluate profanity on a scale of 0.0 (clean) to 1.0 (highly profane) in Nepali cultural context.
> Respond ONLY with a valid JSON object in this exact format:
> {\"devanagari\": \"\<converted text\>\", \"profanity_score\": \<float\>}
> Text: \"\<prepared_text\>\"

**6.3 Batch Prompt (Short Texts)**

Used when processing buffered short texts. Multiple documents per LLM call. Integer indices (0, 1, 2\...) are used as keys in the JSON payload rather than MongoDB \_id values (which are 64-character SHA-256 hashes). Using integer keys keeps the prompt compact and avoids LLM hallucination or truncation of long opaque strings.

The worker maintains a local dict mapping index to \_id for the duration of each batch to map results back after parsing.

> Below is a JSON object mapping integer indices to short Romanized Nepali texts. For each entry, convert the text to Devanagari script and evaluate its profanity on a scale of 0.0 (clean) to 1.0 (highly profane) in Nepali cultural context. Preserve all \[ENG_N\] placeholders exactly as they appear.
> Respond ONLY with a valid JSON object mapping the same integer indices to result objects. Do not include markdown, preamble, or any other text.
> Input:
> {
> \"0\": \"kassam\",
> \"1\": \"wow \[ENG_0\] party\",
> \"2\": \"chus na ta\"
> }
> Expected output format:
> {
> \"0\": {\"devanagari\": \"कसम\", \"profanity_score\": 0.0},
> \"1\": {\"devanagari\": \"वाउ \[ENG_0\] पार्टी\", \"profanity_score\": 0.0},
> \"2\": {\"devanagari\": \"चुस न त\", \"profanity_score\": 0.9}
> }
> Now process this input:
> \<json_payload\>

**7. Error Handling & Resilience**

**7.1 JSON Parse Failure or Validation Failure**

- Catch json.JSONDecodeError and any field validation errors

- Log the document \_id and the raw LLM response string for debugging

- Increment ai_slots.retry_count in MongoDB using \$inc

- Do not re-queue the document in the current run --- it will be picked up on the next worker pass until retry_count reaches MAX_RETRY_COUNT

**7.2 Placeholder Validation Failure**

If any \[ENG_N\] placeholder from the prepared text is absent from the LLM\'s devanagari output, treat this as a validation failure and apply the same handling as a JSON parse failure --- log, increment retry_count, skip.

**7.3 Batch Partial Failure**

In batch mode, the LLM may return results for only some of the indices in the payload. Process only the indices that are present and valid in the response. For each index that is absent or invalid, increment retry_count for the corresponding document individually. Do not fail the entire batch because of one missing entry.

**7.4 Ollama Timeout**

If the HTTP request exceeds LLM_REQUEST_TIMEOUT_SECONDS, log the timeout with the document \_id(s) involved, sleep briefly (e.g. 5 seconds) to allow the GPU to recover, then continue. Affected documents are picked up on the next pass via retry_count.

**7.5 Poison Pill Documents**

Documents where ai_slots.retry_count \>= MAX_RETRY_COUNT are excluded by the fetch query and permanently skipped in all future runs. At worker startup, query and log a count of poisoned documents so the operator has visibility without needing a manual query.

**8. Input / Output Contract**

**8.1 What the Worker Reads**

|                                          |              |                                                        |
|------------------------------------------|--------------|--------------------------------------------------------|
| **Field**                                | **Location** | **Used For**                                           |
| cleaned_text                             | Top-level    | Source text for preparation and passthrough            |
| linguistic_profile.english_token_indices | Top-level    | Masking English tokens before LLM call                 |
| linguistic_profile.has_devanagari        | Top-level    | Short-circuit: skip LLM if already Devanagari          |
| metadata.total_global_occurrences        | Top-level    | Fetch ordering --- high frequency first                |
| ai_slots.devanagari_translation          | ai_slots     | Null check --- determines if document needs processing |
| ai_slots.retry_count                     | ai_slots     | Excludes poisoned documents from fetch                 |

**8.2 What the Worker Writes**

|                                 |                                                          |                                      |
|---------------------------------|----------------------------------------------------------|--------------------------------------|
| **Field**                       | **Value Written**                                        | **Condition**                        |
| ai_slots.devanagari_translation | Rehydrated Devanagari string or cleaned_text passthrough | Always on success                    |
| ai_slots.profanity_score        | Float 0.0--1.0                                           | LLM path only; null for passthroughs |
| ai_slots.model_version          | \"ollama/\<model\>\" or passthrough sentinel             | Always on success                    |
| ai_slots.translated_at          | Unix timestamp (time.time())                             | Always on success                    |
| ai_slots.retry_count            | Incremented by 1 via \$inc                               | On any failure                       |

**9. Key Design Decisions & Rationale**

|                      |                                              |                                        |                                                                                                                                            |
|----------------------|----------------------------------------------|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| **Decision**         | **Chosen Approach**                          | **Rejected Alternative**               | **Reason**                                                                                                                                 |
| Batch keys           | Integer indices 0,1,2\...                    | SHA-256 \_id as key                    | 64-char opaque strings risk LLM truncation or hallucination; integers keep prompts compact and mapping deterministic                       |
| English preservation | Python rehydration of \[ENG_N\] placeholders | Instruct LLM to preserve English words | LLM compliance with word-level instructions is unreliable; Python rehydration is deterministic                                             |
| Routing threshold    | Character count of prepared text             | Token count from metadata              | Character count is a direct LLM context proxy and requires no tokenizer call; token_count in metadata counts Nepali tokens, not LLM tokens |
| Ollama endpoint      | /api/generate with format:json               | /api/chat                              | format:json constrains the output tokenizer to valid JSON natively, reducing parse failures                                                |
| Processing order     | total_global_occurrences descending          | Default cursor order                   | Maximises coverage value per LLM call --- high-frequency terms benefit the most documents first                                            |
| English detection    | wordfreq top-N frequency list                | \_COMMON_ENGLISH_WORDS hardcoded set   | Frequency ranking avoids collision between rare English words and common Romanized Nepali tokens; threshold is configurable                |
| Retry mechanism      | retry_count field + fetch query exclusion    | In-memory dead letter queue            | Persistent retry state survives worker restarts; poisoned documents are visible in MongoDB without code                                    |

**10. Known Limitations**

- English token indices in linguistic_profile are computed by the ETL using \_COMMON_ENGLISH_WORDS only. Less common English words and proper nouns embedded in Nepali text will not be masked before the LLM call. The enrichment worker does not supplement or repair these indices at runtime --- this is accepted and out of scope.

- wordfreq is used in the enrichment worker but not in the ETL pipeline. The two English detection strategies will produce different results for borderline tokens. This inconsistency is accepted until the ETL is updated as a follow-up task.

- Lingua (used in lang_filter.py for document-level language detection) is not used in the enrichment worker. Token-level classification with Lingua is unreliable on single tokens due to insufficient n-gram context. wordfreq lookup is the appropriate tool at this granularity.

- The worker does not reconstruct punctuation that was stripped by the ETL. The devanagari_translation field stores the transliteration of what is in cleaned_text, which already reflects ETL cleaning decisions.
