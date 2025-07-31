## Dependency Graph

Based on the static analysis and runtime tracing, the following files and directories constitute the `KEEP_SET` required for the core pipeline to function:

```
src/pynucleus/cli.py
src/pynucleus/utils/telemetry_patch.py
src/pynucleus/utils/error_handler.py
src/pynucleus/integration/config_manager.py
src/pynucleus/pipeline/pipeline_utils.py
src/pynucleus/pipeline/pipeline_rag.py
src/pynucleus/rag/vector_store.py
src/pynucleus/rag/answer_processing.py
src/pynucleus/llm/answer_engine.py
src/pynucleus/llm/llm_runner.py
src/pynucleus/llm/prompting.py
src/pynucleus/eval/confidence_calibration.py
src/pynucleus/data/mock_data_manager.py
src/pynucleus/pipeline/financial_analyzer.py
src/pynucleus/metrics/system_statistics.py
src/pynucleus/diagnostics/runner.py
src/pynucleus/eval/golden_eval.py
configs/development_config.json
configs/mock_data_modular_plants.json
data/product_prices.json
data/03_intermediate/vector_db/
data/04_models/
data/calibration/models/
logs/
```