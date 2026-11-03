# Genie Space Evaluator

A Databricks Asset Bundle for measuring whether changes to a Genie space
(instructions, example SQL, sample questions) improve benchmark accuracy.

Each developer deploys an isolated copy of the space, job, and MLflow
experiment. A CI job does the same per pull request, fails closed on a
regression, then tears the isolated resources down.

No Databricks Apps and no Lakebase. History lives in MLflow plus Delta tables.

## What lives in this repo

| Path | Role |
| --- | --- |
| [`databricks.yml`](databricks.yml) | Bundle root (`engine: direct`, `dev` + `ci` targets) |
| [`genie_space/eval_space.geniespace.json`](genie_space/eval_space.geniespace.json) | The space under test, including `benchmarks.questions` |
| [`resources/`](resources/) | Genie space, MLflow experiment, eval job, and optimize job |
| [`src/genie_evaluator/`](src/genie_evaluator/) | Eval-Run client, optimizer, MLflow/Delta logging, regression gate |
| [`jobs/run_eval.py`](jobs/run_eval.py) | Eval job notebook |
| [`jobs/run_optimize_baseline.py`](jobs/run_optimize_baseline.py) | Optimize job: template baseline |
| [`jobs/run_optimize_experiment.py`](jobs/run_optimize_experiment.py) | Optimize job: one lever or named experiment on a throwaway clone |
| [`experiments/`](experiments/) | Hand-authored patch sets (`*.json`) |

## One-time setup

1. Install [Databricks CLI](https://docs.databricks.com/dev-tools/cli/databricks-cli.html) >= 1.3.0.
2. Authenticate: `databricks auth login --host https://adb-984752964297111.11.azuredatabricks.net`
3. Replace the sample NYC taxi space with a clone of your production space:

```bash
databricks bundle generate genie-space \
  --existing-id <your-production-space-id> \
  --key eval_space \
  --resource-dir resources \
  --genie-space-dir genie_space \
  --force
```

Then restore the `title` / `description` / `permissions` you want in
[`resources/eval_space.genie_space.yml`](resources/eval_space.genie_space.yml), and add benchmark
Q&A pairs under `benchmarks.questions` in the generated JSON. See
[`genie_space/README.md`](genie_space/README.md).

4. Set `warehouse_id` when you deploy (required):

```bash
databricks bundle deploy --var warehouse_id=<sql-warehouse-id>
```

`dev` is the default target. `mode: development` prefixes every resource
with `[dev <your username>]`, so two developers never share a space, job,
or experiment.

## Day-to-day workflow

1. Edit instructions, example SQL, or benchmarks in
   `genie_space/eval_space.geniespace.json` (or edit in the UI and pull
   changes back with `databricks bundle generate genie-space --resource eval_space --force`).
2. Deploy and run:

```bash
databricks bundle deploy --var warehouse_id=<sql-warehouse-id>
databricks bundle run genie_eval_job --params git_sha=$(git rev-parse HEAD),git_branch=$(git branch --show-current)
```

3. Inspect the result:
   - MLflow experiment `genie-eval-experiment` (prefixed in dev)
   - Delta tables `` `{catalog}.{schema}.genie_eval_runs` `` and
     `` `genie_eval_results` `` — in dev, `schema` is your username under
     catalog `main`

The job fails if:

- accuracy is below `min_accuracy` (default `0.0`), or
- accuracy drops more than `max_allowed_regression` (default `0.02`) versus
  the latest recorded run on `baseline_branch` (default `main`)

The first run has no baseline, so only the absolute floor applies.

## Automated optimization

`genie_opt_job` measures the impact of tweaking different Genie surfaces
without a Databricks App and without mutating the template eval space or
the original source tables/views.

It is **not** wired into CI. Run it by hand from a deployed `dev` target.

```bash
databricks bundle run genie_opt_job \
  --params opt_mode=auto,levers='["descriptions","joins","instructions"]',max_attempts=2
```

`opt_mode` is the switch. Leave experiment JSON in [`experiments/`](experiments/);
**auto mode never reads that folder.**

| `opt_mode` | What runs |
| --- | --- |
| `auto` (default) | LLM lever loop (`levers=`). Files in `experiments/` are ignored. |
| `experiments` | Hand-authored files. `experiments=` names a subset; omit it to run every `experiments/*.json`. |

### Hand-authored experiments

```bash
# every file in experiments/
databricks bundle run genie_opt_job --params opt_mode=experiments

# a subset — the rest of the folder stays put
databricks bundle run genie_opt_job \
  --params opt_mode=experiments,experiments='["synonyms_account_status","column_settings_party_name"]'
```

Each file is one parallel clone:

```json
{
  "name": "synonyms_account_status",
  "hypothesis": "Genie misses informal wording for account status.",
  "patches": [
    {
      "op": "update_column_synonyms",
      "identifier": "sandbox.jg_banking_genie.account",
      "column_name": "status",
      "synonyms": ["open/closed", "account status", "live"]
    }
  ]
}
```

Patch ops you can mix in one file (`lever` is `manual`, so any op is allowed):

| op | What it changes |
| --- | --- |
| `update_table_description` | Table `description` |
| `update_column_description` | Column `description` |
| `update_column_synonyms` | Column `synonyms` |
| `update_column_settings` | Flags: `enable_entity_matching`, `enable_format_assistance`, `get_example_values`, `build_value_dictionary`, `exclude` |
| `update_column_config` | Any mix of description, synonyms, and the flags above |
| `add_sample_question` | UI sample question |
| `add_join_spec` / `update_join_spec` | Preferred joins |
| `update_instruction` | Appends to the single text-instruction block |
| `add_sql_snippet` / `add_example_sql` | Reusable SQL / worked examples (`synonyms` allowed on snippets) |
| `clone_and_comment_source` | Wrapper view + `COMMENT ON` (never mutates the original) |

What it does:

1. Evaluates the bundle-managed eval space as a **read-only template** (baseline).
2. Fans out one throwaway Genie clone per lever (`for_each`, concurrency 4).
   Two `bundle run genie_opt_job` invocations can overlap; they do not share a clone.
3. An LLM (`llm_endpoint`, default `databricks-meta-llama-3-3-70b-instruct`)
   proposes a small patch set for that lever only.
4. Accepts the attempt only if accuracy is strictly higher **and** no
   previously-GOOD question regresses.
5. Always deletes the clone. Wrapper views, when used, live in
   `{catalog}.opt_{run8}_{lever}` and are dropped with `CASCADE`.

Levers: `descriptions`, `joins`, `instructions`, `sql_snippets`, `source_copies`.

`source_copies` never `CREATE OR REPLACE`s originals. It wraps each selected
source as `SELECT * FROM original` in the experiment schema (so SQL snippet
table names still match), applies `COMMENT ON` there, and retargets only
`data_sources` / join identifiers on the **clone**. Benchmark expected SQL
keeps pointing at the originals.

What is **not** touched:

- The production Genie space
- The bundle eval space config (champions are not written back)
- `sandbox.jg_banking_genie.*` (or whatever the template lists as sources)
- Benchmark definitions

Read the impact in the same MLflow experiment:

- Parent run `opt-baseline-*` has `template_serialized_space.json`
- One run per attempt, tagged `lever` / `decision`, with `accuracy_delta`,
  `patches.json`, and `serialized_space.json` (the config that actually scored)

Delta: `genie_opt_runs` and `genie_opt_iterations` in `{catalog}.{schema}`,
plus the usual `genie_eval_runs` / `genie_eval_results` rows
(`triggered_by=optimize:<lever>`).

To copy a winning config into git after you inspect it in MLflow, download
`serialized_space.json` and merge it into
`genie_space/eval_space.geniespace.json` yourself — the job will not do that.

## CI

[`.github/workflows/genie_eval_ci.yml`](.github/workflows/genie_eval_ci.yml)
on pull requests:

1. `databricks bundle deploy -t ci --var pr_number=$PR`
2. `databricks bundle run genie_eval_job -t ci` — non-zero exit fails the check
3. `databricks bundle destroy -t ci` always, so PR copies do not leak

On push to `main` it re-runs once so later PRs have a recorded baseline.

Repo secrets:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN` (CI service principal)
- `GENIE_WAREHOUSE_ID`

## Textual answers (later)

The native Eval-Run API grades SQL only. Conversation-API grading of Genie's
prose answers is intentionally not in this first version.
