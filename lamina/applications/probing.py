"""
Linear probing pipeline for lamina InternalsRecord collections.

Trains a logistic regression (linear probe) per ``(layer, property)`` pair
and evaluates how well each layer's representations encode a target property.

Quick-start
-----------
::

    from lamina.applications.probing import LinearProbe

    records = dataset.run(model, tokenizer)
    probe   = LinearProbe()
    results = probe.fit_all(
        records,
        label_fn=lambda r: r.properties["pos"],
        source="input_mean",
    )
    for r in results:
        print(f"layer {r.layer:2d}  ({r.normalized_depth:.2f})  acc={r.accuracy:.3f}")

Normalized depth
----------------
``ProbeResult.normalized_depth = layer / (num_layers - 1)`` enables
cross-architecture comparison between models with different depths (e.g.
GPT-2 vs Llama-3 vs Mamba).  Layer 0 (embedding) is depth 0.0; the final
layer is depth 1.0.

Conditions
----------
Pass ``condition_fn`` to split records by condition and train separate probes
per condition.  Results will have ``condition`` set to the condition key::

    results = probe.fit_all(
        records,
        label_fn=lambda r: r.properties["pos"],
        condition_fn=lambda r: r.properties.get("domain"),
        source="input_mean",
    )

Dependencies
------------
Requires ``scikit-learn``::

    pip install scikit-learn
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# ProbeResult
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """
    Evaluation result for one ``(layer, property[, condition])`` probe.

    Attributes
    ----------
    layer : int
        0-based layer index.  Layer 0 = embedding output.
    property_name : str
        Name of the target property (e.g. ``"pos"``, ``"dep"``).
    condition : str | None
        Condition key when ``condition_fn`` was supplied to
        :meth:`LinearProbe.fit_all`; ``None`` otherwise.
    accuracy : float
        Fraction of correctly classified test examples.
    f1 : float
        Macro-averaged F1 score across classes.
    n_train : int
        Number of training examples.
    n_test : int
        Number of test examples.
    normalized_depth : float
        ``layer / (num_layers - 1)`` — enables cross-architecture comparison.
        Layer 0 (embedding) = 0.0; final layer = 1.0.
    classes : list[str]
        Sorted class labels used in training.
    """
    layer: int
    property_name: str
    condition: Optional[str]
    accuracy: float
    f1: float
    n_train: int
    n_test: int
    normalized_depth: float
    classes: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        cond = f", cond={self.condition!r}" if self.condition is not None else ""
        return (
            f"ProbeResult(layer={self.layer}, depth={self.normalized_depth:.2f}"
            f"{cond}, acc={self.accuracy:.3f}, f1={self.f1:.3f}"
            f", n={self.n_train}+{self.n_test})"
        )


# ---------------------------------------------------------------------------
# Hidden-state source helpers
# ---------------------------------------------------------------------------

def _layer_vector(
    record: Any,
    source: Union[str, Callable],
    layer: int,
) -> Optional[np.ndarray]:
    """
    Extract the ``(hidden_dim,)`` feature vector for *layer* from *record*.

    *source* is one of:

    ``"input_mean"``
        ``record.run.input_hidden_states_mean[layer, 0, :]``
    ``"output_mean"``
        ``record.run.output_hidden_states_mean[layer, 0, :]``
    ``"last_output"``
        ``record.run.last_output_hidden_state[layer, :]``
    ``"thinking"``
        ``record.thinking_hidden_state[layer, :]``
    ``"span:<name>"``
        ``record.span_hidden_states_mean["<name>"][layer, :]``
    callable
        ``source(record, layer)`` → ``np.ndarray`` of shape ``(hidden_dim,)``

    Returns ``None`` if the requested data is not available on the record.
    """
    if callable(source):
        return source(record, layer)

    run = record.run

    if source == "input_mean":
        m = getattr(run, "input_hidden_states_mean", None)
        if m is None or layer >= m.shape[0]:
            return None
        return m[layer, 0, :].astype(np.float32)

    if source == "output_mean":
        m = getattr(run, "output_hidden_states_mean", None)
        if m is None or layer >= m.shape[0]:
            return None
        return m[layer, 0, :].astype(np.float32)

    if source == "last_output":
        m = getattr(run, "last_output_hidden_state", None)
        if m is None or layer >= m.shape[0]:
            return None
        return m[layer, :].astype(np.float32)

    if source == "thinking":
        m = getattr(record, "thinking_hidden_state", None)
        if m is None or layer >= m.shape[0]:
            return None
        return m[layer, :].astype(np.float32)

    if source.startswith("span:"):
        span_name = source[len("span:"):]
        spans = getattr(record, "span_hidden_states_mean", None) or {}
        m = spans.get(span_name)
        if m is None or layer >= m.shape[0]:
            return None
        return m[layer, :].astype(np.float32)

    raise ValueError(
        f"Unknown probe source {source!r}.  "
        "Expected 'input_mean', 'output_mean', 'last_output', 'thinking', "
        "'span:<name>', or a callable (record, layer) → np.ndarray."
    )


def _num_layers_from_records(
    records: List[Any],
    source: Union[str, Callable],
) -> int:
    """Infer the number of layers from the first record that has data."""
    for rec in records:
        run = rec.run
        if source == "input_mean":
            m = getattr(run, "input_hidden_states_mean", None)
            if m is not None:
                return m.shape[0]
        elif source == "output_mean":
            m = getattr(run, "output_hidden_states_mean", None)
            if m is not None:
                return m.shape[0]
        elif source in ("last_output", "thinking"):
            attr = ("last_output_hidden_state" if source == "last_output"
                    else "thinking_hidden_state")
            m = getattr(run, attr, None)
            if m is None:
                m = getattr(rec, attr, None)
            if m is not None:
                return m.shape[0]
        elif source.startswith("span:"):
            spans = getattr(rec, "span_hidden_states_mean", None) or {}
            m = spans.get(source[len("span:"):])
            if m is not None:
                return m.shape[0]
        elif callable(source):
            try:
                v = source(rec, 0)
                if v is not None:
                    return getattr(run, "num_layers", None) or 1
            except Exception:
                pass
    raise ValueError(
        "Could not determine num_layers from records.  "
        "Ensure at least one record has hidden states extracted for "
        f"source={source!r}."
    )


# ---------------------------------------------------------------------------
# LinearProbe
# ---------------------------------------------------------------------------

class LinearProbe:
    """
    Train and evaluate a logistic regression per ``(layer, property)`` pair.

    Parameters
    ----------
    C : float
        Regularisation strength for ``sklearn.linear_model.LogisticRegression``
        (inverse regularisation — larger = less regularisation).
    max_iter : int
        Maximum iterations for the solver.
    test_frac : float
        Fraction of examples held out for evaluation (stratified split).
    random_state : int
        Random seed for reproducibility.
    normalize : bool
        When ``True``, standardise features to zero mean and unit variance per
        layer before fitting.  Recommended when comparing across layers.

    Example
    -------
    ::

        probe = LinearProbe(C=1.0, test_frac=0.2, normalize=True)
        results = probe.fit_all(
            records,
            label_fn=lambda r: r.properties["pos"],
            source="input_mean",
            property_name="POS tag",
        )
        # Sort by depth for a nice layer-by-layer view
        for r in sorted(results, key=lambda r: r.layer):
            print(f"layer {r.layer:3d}  depth={r.normalized_depth:.2f}"
                  f"  acc={r.accuracy:.3f}  f1={r.f1:.3f}")
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 200,
        test_frac: float = 0.2,
        random_state: int = 42,
        normalize: bool = True,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.test_frac = test_frac
        self.random_state = random_state
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_all(
        self,
        records: List[Any],
        label_fn: Callable[[Any], Any],
        source: Union[str, Callable] = "input_mean",
        property_name: str = "property",
        condition_fn: Optional[Callable[[Any], Any]] = None,
        layers: Optional[List[int]] = None,
    ) -> List[ProbeResult]:
        """
        Train and evaluate a probe for every layer.

        Parameters
        ----------
        records : list[InternalsRecord]
        label_fn : callable(record) → label
            Extract the string (or int) class label from a record.
            Records where ``label_fn`` returns ``None`` are skipped.
        source : str | callable
            Which hidden state to use as features.  See
            :func:`_layer_vector` for accepted values.
        property_name : str
            Human-readable name stored in :attr:`ProbeResult.property_name`.
        condition_fn : callable(record) → str | None, optional
            When supplied, records are grouped by the return value and a
            separate probe is trained per group.  Records where this function
            returns ``None`` are skipped.
        layers : list[int] | None
            If supplied, only probe these layer indices.  Otherwise all layers
            are probed.

        Returns
        -------
        list[ProbeResult]
            One entry per ``(layer[, condition])``, sorted by
            ``(condition, layer)``.
        """
        _require_sklearn()

        num_layers = _num_layers_from_records(records, source)
        probe_layers = layers if layers is not None else list(range(num_layers))

        if condition_fn is None:
            groups: Dict[Optional[str], List[Any]] = {None: list(records)}
        else:
            groups = {}
            for rec in records:
                cond = condition_fn(rec)
                if cond is None:
                    continue
                groups.setdefault(str(cond), []).append(rec)

        results: List[ProbeResult] = []
        for condition, group_records in groups.items():
            group_results = self._fit_group(
                group_records, label_fn, source,
                property_name, condition, probe_layers, num_layers,
            )
            results.extend(group_results)

        results.sort(key=lambda r: (r.condition or "", r.layer))
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fit_group(
        self,
        records: List[Any],
        label_fn: Callable,
        source: Union[str, Callable],
        property_name: str,
        condition: Optional[str],
        layers: List[int],
        num_layers: int,
    ) -> List["ProbeResult"]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        # Filter records with valid labels
        filtered = [
            (rec, label_fn(rec))
            for rec in records
        ]
        filtered = [(rec, lab) for rec, lab in filtered if lab is not None]
        if len(filtered) < 2:
            return []

        recs, raw_labels = zip(*filtered)
        le = LabelEncoder()
        y  = le.fit_transform(raw_labels)
        classes = list(le.classes_)

        results: List[ProbeResult] = []

        for layer in layers:
            # Build feature matrix for this layer
            X_rows: List[np.ndarray] = []
            y_rows: List[int] = []
            for rec, yi in zip(recs, y):
                vec = _layer_vector(rec, source, layer)
                if vec is None:
                    continue
                X_rows.append(vec)
                y_rows.append(int(yi))

            if len(X_rows) < 2 or len(set(y_rows)) < 2:
                continue

            X = np.stack(X_rows, axis=0)
            y_arr = np.array(y_rows, dtype=np.int32)

            # Stratified split
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y_arr,
                    test_size=self.test_frac,
                    random_state=self.random_state,
                    stratify=y_arr,
                )
            except ValueError:
                # Not enough samples in some class for stratify
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y_arr,
                    test_size=self.test_frac,
                    random_state=self.random_state,
                )

            # Optional standardisation
            if self.normalize:
                scaler  = StandardScaler()
                X_tr    = scaler.fit_transform(X_tr)
                X_te    = scaler.transform(X_te)

            # Fit
            clf = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                multi_class="auto",
                solver="lbfgs",
            )
            clf.fit(X_tr, y_tr)

            # Evaluate
            y_pred   = clf.predict(X_te)
            acc      = float(accuracy_score(y_te, y_pred))
            f1       = float(f1_score(
                y_te, y_pred,
                average="macro",
                zero_division=0,
            ))
            norm_dep = layer / max(num_layers - 1, 1)

            results.append(ProbeResult(
                layer=layer,
                property_name=property_name,
                condition=condition,
                accuracy=acc,
                f1=f1,
                n_train=len(y_tr),
                n_test=len(y_te),
                normalized_depth=norm_dep,
                classes=classes,
            ))

        return results


# ---------------------------------------------------------------------------
# Normalized-depth utility (standalone, no sklearn needed)
# ---------------------------------------------------------------------------

def normalized_depth(layer: int, num_layers: int) -> float:
    """
    Map *layer* to ``[0, 1]`` for cross-architecture comparison.

    ``layer=0`` (embedding) maps to ``0.0``; ``layer=num_layers-1``
    (final transformer block) maps to ``1.0``.

    Parameters
    ----------
    layer : int
        0-based layer index.
    num_layers : int
        Total number of layers (embedding + all transformer blocks).

    Returns
    -------
    float in ``[0, 1]``
    """
    if num_layers <= 1:
        return 0.0
    return layer / (num_layers - 1)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "lamina.applications.probing requires scikit-learn.\n"
            "Install it with:  pip install scikit-learn"
        ) from exc
