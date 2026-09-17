"""MLflow tracking for experiments: parameters, metrics, artifacts, provenance.

Why. An experiment is only reproducible if everything that determines its
result is recorded: the exact command, the seeds, the input data *by content*
rather than by path, the model weights by content hash, the code revision and
the versions of the libraries that do the arithmetic. A file name does not
identify a dataset (the same name holds different pools over time) and a path
does not identify a model (weights are replaced in place), so both are logged
by sha256.

Tracking must never break an experiment. A missing package, an unreachable
tracking server or an error inside the tracker is logged and swallowed; the
experiment keeps running. Computation is the job, logging is the bookkeeping.

Entry point: :class:`RunTracker`. One instance per run::

    tracker = RunTracker(experiment="hyperactive-training")
    with tracker.run("demo", tags={"run_type": "train"}) as run:
        run.log_environment()
        run.log_params({"seed": 1})
        run.log_dataset("data/synthetic/wells.csv", "wells")
        run.log_metrics({"npv_rl": 1.0}, step=1)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# MLflow caps parameter values; lists of constraints and other nested
# structures do not fit. Scalars go in as parameters, everything else is
# attached as a JSON artifact.
PARAM_VALUE_LIMIT = 480

DEFAULT_EXPERIMENT = "hyperactive"
# A local file store by default: reproducible tracking must not require a
# server to be running. Point HYPERACTIVE_MLFLOW_TRACKING_URI at an MLflow
# server (see deploy/mlflow) to collect runs centrally instead.
DEFAULT_TRACKING_URI = "file:./mlruns"

TRACKING_URI_ENV = "HYPERACTIVE_MLFLOW_TRACKING_URI"
EXPERIMENT_ENV = "HYPERACTIVE_MLFLOW_EXPERIMENT"
ENABLED_ENV = "HYPERACTIVE_MLFLOW_ENABLED"

# MLflow accepts only latin letters, digits and _ - . : / and space in
# parameter names. Settings contain dictionaries with Cyrillic keys - the
# drilling cost per metre by well type, "ГС+ГРП". Such a name rejects the whole
# batch, so Cyrillic is transliterated and anything else disallowed becomes an
# underscore.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.: /")


def sha256_of(path: str | Path) -> str:
    """Hash of a file's content: this is what identifies a dataset or weights."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_param_key(key: str) -> str:
    """A parameter name MLflow accepts, keeping it readable."""
    out = []
    for char in str(key):
        lowered = char.lower()
        if lowered in _TRANSLIT:
            translit = _TRANSLIT[lowered]
            out.append(translit.upper() if char.isupper() else translit)
        elif char in _ALLOWED:
            out.append(char)
        else:
            out.append("_")
    cleaned = "".join(out)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "param"


def _flatten(prefix: str, value: Any) -> dict[str, str]:
    """Scalar fields become parameters; nested structures go in as artifacts."""
    out: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(_flatten(f"{prefix}.{key}" if prefix else str(key), item))
    elif isinstance(value, (list, tuple)):
        out[prefix] = ",".join(str(item) for item in value)[:PARAM_VALUE_LIMIT] or "[]"
    elif value is not None:
        text = str(value)
        out[prefix] = text if len(text) <= PARAM_VALUE_LIMIT else text[:PARAM_VALUE_LIMIT] + "..."
    return out


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "n/a"


def git_revision(path: str | Path | None = None) -> dict[str, str]:
    """Commit and dirty flag of the repository the code runs from.

    Without it a run records which parameters produced a number but not which
    code did. Returns an empty dict outside a repository.
    """
    cwd = Path(path) if path is not None else Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, timeout=5)
        if sha.returncode != 0:
            return {}
        status = subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                                capture_output=True, text=True, timeout=5)
        return {"git_sha": sha.stdout.strip(),
                "git_dirty": "true" if status.stdout.strip() else "false"}
    except (OSError, subprocess.SubprocessError):
        return {}


def environment_info() -> dict[str, Any]:
    """Everything about the runtime that can change a number.

    Deliberately excludes hostname and user: a run must identify the software,
    not the machine or the person who launched it.
    """
    version = _package_version("hyperactive")
    if version == "n/a":
        # Running from a source checkout rather than an installed distribution.
        try:
            from hyperactive import __version__ as version
        except Exception:
            version = "n/a"
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "hyperactive": version,
    }
    for package in ("numpy", "pandas", "torch", "stable_baselines3", "gymnasium",
                    "pydantic", "mlflow"):
        info[package] = _package_version(package.replace("_", "-"))
    info.update(git_revision())
    return info


class RunTracker:
    """One MLflow run: parameters, metrics, artifacts.

    Every method is wrapped so that a tracking failure cannot affect the
    experiment. When tracking is disabled or MLflow is unavailable the tracker
    stays usable and does nothing.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or os.environ.get(
            TRACKING_URI_ENV, DEFAULT_TRACKING_URI)
        self.experiment = experiment or os.environ.get(EXPERIMENT_ENV, DEFAULT_EXPERIMENT)
        if enabled is None:
            raw = os.environ.get(ENABLED_ENV, "true")
            enabled = raw.strip().lower() not in {"0", "false", "no", "off"}
        self.enabled = bool(enabled)
        self._mlflow: Any = None
        self._run: Any = None
        self._started = 0.0

    # ------------------------------------------------------------------ setup
    def _client(self) -> Any:
        """The package is imported lazily: experiments must run without it."""
        if self._mlflow is None:
            # The default client waits two minutes and retries five times: with
            # a dead server the experiment would stall on logging. The ceiling
            # is short but overridable from outside.
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
            import mlflow  # local import on purpose

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment)
            self._mlflow = mlflow
        return self._mlflow

    @property
    def run_id(self) -> str:
        return self._run.info.run_id if self._run is not None else ""

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None,
            nested: bool = False, log_duration: bool = True) -> Iterator["RunTracker"]:
        """Open a run; on any tracking failure simply switch tracking off.

        ``nested=True`` opens a child run of the currently active one - this is
        how a benchmark suite holds one run per grid cell under the suite run.

        ``log_duration=False`` for a run opened after the work it describes has
        already finished: wall-clock time of the logging call is not the
        duration of the computation and would read as one.
        """
        if not self.enabled:
            yield self
            return
        started = time.monotonic()
        previous_run = self._run
        try:
            mlflow = self._client()
            self._run = mlflow.start_run(run_name=name, tags=tags or {}, nested=nested)
        except Exception as exc:
            logger.warning("MLflow is unavailable (%s), running without tracking: %s",
                           self.tracking_uri, exc)
            self.enabled = False
            self._run = previous_run
            yield self
            return
        self._started = started
        try:
            yield self
        finally:
            if log_duration:
                self._safe(lambda: self._mlflow.log_metric(
                    "duration_seconds", round(time.monotonic() - started, 2)))
            self._safe(self._mlflow.end_run)
            self._run = previous_run

    def _safe(self, action) -> None:
        """A tracking error must not escape and break the experiment."""
        if not self.enabled:
            return
        try:
            action()
        except Exception as exc:
            logger.warning("MLflow: could not write to the run: %s", exc)

    # ----------------------------------------------------------------- writing
    def log_params(self, params: dict[str, Any]) -> None:
        """Writes parameters as a batch, falling back to one by one.

        A single invalid name rejects the whole batch, and all the settings go
        with it. Retrying individually keeps everything the server accepts and
        names the broken one in the log.
        """
        clean = {safe_param_key(k): v for k, v in _flatten("", params).items() if k}
        if not clean or not self.enabled:
            return
        try:
            self._mlflow.log_params(clean)
            return
        except Exception as exc:
            logger.warning("MLflow: parameter batch rejected (%s), writing one by one", exc)
        for key, value in clean.items():
            try:
                self._mlflow.log_param(key, value)
            except Exception as exc:
                logger.warning("MLflow: parameter %s not written: %s", key, exc)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Numeric values only; ``step`` turns them into a series over episodes."""
        numeric = {}
        for key, value in metrics.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number or number in (float("inf"), float("-inf")):
                continue  # NaN and infinities are not metrics
            numeric[safe_param_key(key)] = number
        if numeric:
            self._safe(lambda: self._mlflow.log_metrics(numeric, step=step))

    def log_tags(self, tags: dict[str, str]) -> None:
        clean = {safe_param_key(k): str(v)[:PARAM_VALUE_LIMIT]
                 for k, v in tags.items() if v is not None}
        if clean:
            self._safe(lambda: self._mlflow.set_tags(clean))

    def log_dict(self, payload: Any, filename: str) -> None:
        self._safe(lambda: self._mlflow.log_dict(payload, filename))

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        source = Path(path)
        if not source.exists():
            logger.warning("MLflow: artifact not found, skipping: %s", source)
            return
        self._safe(lambda: self._mlflow.log_artifact(str(source), artifact_path=artifact_path))

    def log_json_artifact(self, payload: Any, filename: str,
                          artifact_path: str = "outputs") -> None:
        """Large results are written as files, not as parameters."""
        def action() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / filename
                target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                  encoding="utf-8")
                self._mlflow.log_artifact(str(target), artifact_path=artifact_path)

        self._safe(action)

    def log_gzip_artifact(self, payload: Any, filename: str,
                          artifact_path: str = "outputs") -> None:
        """A large attachment compressed: a plan with production profiles is megabytes."""
        import gzip

        def action() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / filename
                data = json.dumps(payload, ensure_ascii=False, default=str)
                with gzip.open(target, "wt", encoding="utf-8") as handle:
                    handle.write(data)
                self._mlflow.log_artifact(str(target), artifact_path=artifact_path)

        self._safe(action)

    def log_dataset(self, path: str | Path, input_type: str,
                    filename: str | None = None, attach: bool = False) -> str:
        """An input file together with the hash of its content.

        The hash is the dataset identity: file names repeat from run to run
        while the content differs. Returns the digest so the caller can put it
        into a manifest of its own.

        ``attach=False`` by default: well pools are inputs to be *identified*,
        not copies to be stored in every run. Pass ``attach=True`` for small
        files worth keeping inside the run.
        """
        source = Path(path)
        if not source.exists():
            logger.warning("MLflow: input file not found, skipping: %s", source)
            return ""
        name = filename or source.name
        digest = sha256_of(source)
        self.log_params({f"dataset.{input_type}.filename": name,
                         f"dataset.{input_type}.sha256": digest,
                         f"dataset.{input_type}.bytes": source.stat().st_size})
        if attach:
            self.log_artifact(source, artifact_path="inputs")
        self._log_input(source, input_type, name, digest)
        return digest

    def _log_input(self, source: Path, input_type: str, filename: str, digest: str) -> None:
        """Mark the file as an input of the run, not merely an attachment.

        An attachment is identified by name, and names repeat. An input is
        identified by digest, so runs on the same well pool are found by a
        query instead of by comparing file names.
        """
        def action() -> None:
            from mlflow.entities import Dataset, DatasetInput, InputTag
            from mlflow.tracking import MlflowClient

            dataset = Dataset(
                name=filename,
                # The digest is the version of the input: the full sha256 sits
                # next to it as a parameter, a short one is enough to tell
                # datasets apart here.
                digest=digest[:16],
                source_type="local",
                source=json.dumps({"filename": filename, "sha256": digest}, ensure_ascii=False),
            )
            MlflowClient(tracking_uri=self.tracking_uri).log_inputs(
                self._run.info.run_id,
                [DatasetInput(dataset=dataset,
                              tags=[InputTag("mlflow.data.context", input_type)])],
            )

        self._safe(action)

    # --------------------------------------------------------- reproducibility
    def log_environment(self, extra: dict[str, Any] | None = None) -> None:
        """Command, code revision and library versions - the run's provenance.

        Without this a run says which parameters produced a number but not
        which code and which library versions did the arithmetic, and the
        number cannot be reproduced later.
        """
        info = environment_info()
        command = shlex.join(sys.argv)
        self.log_params({"env": info,
                         "run.command": command,
                         "run.cwd": str(Path.cwd())})
        tags = {"code.git_sha": info.get("git_sha", ""),
                "code.git_dirty": info.get("git_dirty", "")}
        self.log_tags({k: v for k, v in tags.items() if v})
        self.log_dict({"command": sys.argv, "command_line": command,
                       "cwd": str(Path.cwd()), "environment": info,
                       **(extra or {})}, "environment.json")

    def log_seeds(self, **seeds: Any) -> None:
        """Seeds of every generator the result depends on."""
        self.log_params({f"seed.{name}": value for name, value in seeds.items()
                         if value is not None})

    def log_model_artifacts(self, model_dir: str | Path) -> None:
        """A trained model with its manifest, identified by content hash.

        The weights file is attached and its sha256 is logged as a parameter
        and a tag: a path does not identify weights, because they are replaced
        in place under the same name.
        """
        directory = Path(model_dir)
        if not directory.is_dir():
            logger.warning("MLflow: model directory not found, skipping: %s", directory)
            return
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except ValueError:
                manifest = {}
            self.log_params({"model": {k: v for k, v in manifest.items()
                                       if not isinstance(v, (dict, list))}})
            files = manifest.get("files") or {}
            for role, entry in files.items():
                if isinstance(entry, dict) and entry.get("sha256"):
                    # "model" as a role would read as model.model.sha256.
                    name = "weights" if role == "model" else role
                    self.log_params({f"model.{name}.sha256": entry["sha256"]})
            self.log_tags({"model.name": str(manifest.get("name", directory.name)),
                           "model.feature_set": str(manifest.get("feature_set", "")),
                           "model.sha256": str((files.get("model") or {}).get("sha256", ""))})
        for item in sorted(directory.iterdir()):
            if item.is_file():
                self.log_artifact(item, artifact_path="model")
