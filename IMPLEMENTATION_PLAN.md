# Native VOICEVOX Training Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement the independently testable tasks below; the parent integrates the CLI and data stages.

**Goal:** Provide working dataset tools, pinned upstream adapters, genuine train/export/package commands, and explicit compatibility checks without implying a new voice has been trained.

**Architecture:** Preserve the author's networks in pinned source snapshots. A thin Python package manages corpus/features and invokes the actual upstream trainers; export reuses their wrappers and produces the Core 0.17.0 talk signature.

**Tech Stack:** Python 3.11+, NumPy, PyYAML, SoundFile, SciPy; optional torch/librosa/pyworld/pyopenjtalk/ONNX/Core in separate environments.

**Spec:** [DESIGN.md](DESIGN.md), [DATA_FORMAT.md](DATA_FORMAT.md), [BASE_MODELS.md](BASE_MODELS.md).

## Global constraints

- Keep source/service changes inside new_vv_train; reuse existing Core/dictionary/Runtime by path.
- Core 0.17.0, VVM format 2, talk only, sample rate 24000, hop 256, log-F0 with unvoiced 0.
- No downloaded product weights; author source snapshots only. Missing corpus/base cannot count as successful training.
- Each command fails on invalid inputs and records actual outputs; dry-run never marks training complete.
- Parent owns common.py, CLI and data/features; source/training worker owns upstream.py/training.py/config generation; export worker owns export.py/vvm.py/core_validation.py. Workers add separate tests.

## Task 1: Corpus and label contracts

- [x] Write failing tests for repeated audio across splits, path escape, empty WAV, mismatched alignment, natural-log F0, SamplingData conversion.
- [x] Implement common.load_project(path)->dict with absolute _root; common.resolve(cfg,path)->Path; common.write_json(path,value), common.sha256(path); corpus.validate_corpus(cfg)->dict and prepare_audio(cfg)->dict.
- [x] Implement frontend(cfg), import_alignment(cfg) and extract_features(cfg) with profile/speaker/phoneme validation.
- [x] Verify synthetic WAV fixtures and negative cases. No quality claims from generated tone audio.

## Task 2: Author sources, native configs and training

- [x] Write failing tests for locked revision mismatch, native speaker mapping, actual subprocess failure, missing checkpoint and no-success dry-run.
- [x] Implement upstream.fetch_sources(cfg)->dict / inspect_sources(cfg)->dict. Preserve licenses; source snapshots at vendor/<name> with revision and file integrity receipt.
- [x] Implement training.materialize(cfg)->dict, training.train(cfg,module,dry_run=False)->dict, training.validate_base(cfg)->dict; native configs parse under the author's dataclasses.
- [x] Supply configs/module overrides and reviewed patches for API incompatibilities; report unverified GPU dependencies explicitly.

## Task 3: Export and VVM/Core validation

- [x] Write failing tests for mismatched ONNX signature, packaging missing files/duplicate public IDs, and empty or invalid Core output.
- [x] Implement export.export_models(cfg)->dict / validate_onnx(cfg)->dict using locked author's wrappers; merged decode returns wave only. Exercise dynamic lengths.
- [x] Implement vvm.package(cfg)->dict / inspect_vvm(path)->dict; enforce profile and source export receipt, explicit UUID/style mappings.
- [x] Implement core_validation.validate_core(cfg)->dict and validate_service(cfg)->dict using configurable isolated Python interpreter and existing assets, without modifying service.

## Task 4: CLI, docs and integration

- [x] CLI commands: doctor, fetch-sources, validate-corpus, prepare-audio, frontend, import-alignment, extract-features, materialize, validate-base, train, export, validate-onnx, package, inspect-vvm, validate-core, validate-service.
- [x] Exercise editable install, --help, doctor with missing real corpus, and data stages on generated fixtures.
- [x] Run full tests and compile/import checks. Update execution guide and limitations. Full target-voice training remains pending real recordings and a CUDA machine.

## Progress

- Ruling: Workspace is not a Git repository; work directly in the user-requested subdirectory, preserving existing services. No branch/commit is fabricated.
- Ruling: No target recordings or useful pretrained base is supplied; verify tool behavior and source integration, never manufacture a successful voice-training result.
