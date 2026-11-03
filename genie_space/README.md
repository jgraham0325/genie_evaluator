# `eval_space.geniespace.json`

This file is the serialized definition of the Genie space under test:
data sources, instructions, example SQL, sample questions, and — in the
`benchmarks.questions` block — the question/expected-SQL pairs the
evaluation job grades against.

This file currently holds a clone of **JG Banking – Client 360 (synthetic data)**
(`01f1d45d2bcb13f7a512c2a90a102ded`) from field-eng-east, including its
benchmark questions. To point the bundle at a different source space:

```bash
databricks bundle generate genie-space \
  --existing-id <your-production-space-id> \
  --key eval_space \
  --resource-dir resources \
  --genie-space-dir genie_space \
  --force
```

This overwrites `resources/genie_space.yml` and
`genie_space/eval_space.geniespace.json` with the real space's config. After
generating:

1. Re-apply (or merge in) the `permissions`, `title`, and `description`
   overrides from the original `resources/genie_space.yml` if you want to
   keep them — `generate` writes a plain resource block.
2. Add or edit `benchmarks.questions` — these usually are **not** present on
   a production space and are what the eval job actually grades. Each entry
   needs a unique 32-character lowercase-hex `id` (`python3 -c "import uuid;
   print(uuid.uuid4().hex)"`) and exactly one SQL answer:
   ```json
   {
     "id": "<32-hex-chars>",
     "question": ["<the question text>"],
     "answer": [{"format": "SQL", "content": ["<expected SQL>\n"]}]
   }
   ```
   `benchmarks.questions` must stay sorted by `id`.
3. Commit the updated files. From here on, every change to instructions,
   example SQL, or the benchmark suite is a normal, reviewable diff to this
   file, and `databricks bundle deploy` + `databricks bundle run
   genie_eval_job` is how you find out whether it helped.

## Pulling UI edits back into the bundle

If someone edits the deployed space directly in the Databricks UI (common
while iterating), sync those changes back into this file instead of losing
them on the next `bundle deploy`:

```bash
databricks bundle generate genie-space --resource eval_space --force
# or, to keep watching and auto-syncing while you work in the UI:
databricks bundle generate genie-space --resource eval_space --watch --force
```
