"""Delivery-compliance: meet audio tegen broadcast- en streaming-afleverspecs.

Elke spec combineert een loudness-doel (integrated of dialogue-gated), een
true-peak-plafond en eventuele extra's (LRA, RMS-venster, ruisvloer) met de
universele technische QC-poorten (clipping, dropouts, dood kanaal, tegenfase).
`check()` geeft een pass/fail-rapport per criterium; `master_for` in de server
mastert er naartoe en hercontroleert.

Kanttekening bij dialogue-gated specs (Netflix): het officiële meetprotocol
gebruikt Dolby Dialogue Intelligence; wij benaderen dat met BS.1770-loudness
over de gedetecteerde spraaksegmenten. Goed genoeg om te sturen en te
signaleren; de eindcontrole bij de distributeur blijft leidend.
"""

from __future__ import annotations

import numpy as np

SPECS: dict[str, dict] = {
    "ebu-r128": {
        "name": "EBU R128 (Europese omroep)",
        "gating": "integrated",
        "loudness": {"target": -23.0, "tol": 0.5},
        "true_peak_max": -1.0,
        "lra_max_advisory": 20.0,
    },
    "atsc-a85": {
        "name": "ATSC A/85 (Amerikaanse tv)",
        "gating": "integrated",
        "loudness": {"target": -24.0, "tol": 2.0},
        "true_peak_max": -2.0,
    },
    "netflix-2.0": {
        "name": "Netflix non-theatrical 2.0 (dialogue-gated)",
        "gating": "dialogue",
        "loudness": {"target": -27.0, "tol": 2.0},
        "true_peak_max": -2.0,
        # Netflix-leveringsspec: 48 kHz / 24-bit PCM wav
        "format": {"sample_rate": 48000, "min_bit_depth": 24, "pcm": True},
    },
    "apple-podcast": {
        "name": "Apple Podcasts",
        "gating": "integrated",
        "loudness": {"target": -16.0, "tol": 1.0},
        "true_peak_max": -1.0,
    },
    "spotify": {
        "name": "Spotify (loudness-normalisatie-doel)",
        "gating": "integrated",
        "loudness": {"target": -14.0, "tol": 1.0},
        "true_peak_max": -1.0,
    },
    "youtube": {
        "name": "YouTube (loudness-normalisatie-doel)",
        "gating": "integrated",
        "loudness": {"target": -14.0, "tol": 1.0},
        "true_peak_max": -1.0,
    },
    "acx-audiobook": {
        "name": "ACX / Audible audiobook",
        "gating": "integrated",
        "rms_range": (-23.0, -18.0),
        "sample_peak_max": -3.0,
        "noise_floor_max": -60.0,
    },
}


def list_specs() -> list[dict]:
    return [{"id": sid, "name": s["name"], "gating": s.get("gating", "integrated"),
             "loudness_target": (s.get("loudness") or {}).get("target"),
             "true_peak_max": s.get("true_peak_max")}
            for sid, s in SPECS.items()]


def dialogue_loudness(x: np.ndarray, sr: int, segments: list[dict]) -> float | None:
    """BS.1770-loudness over alleen de spraaksegmenten (dialogue-gated benadering)."""
    from chat_with_audio.analysis import measure_lufs

    x2 = x[None, :] if x.ndim == 1 else x
    parts = [x2[:, int(s["start_s"] * sr):int(s["end_s"] * sr)]
             for s in segments if s["kind"] == "speech"]
    if not parts:
        return None
    speech = np.concatenate(parts, axis=1)
    if speech.shape[1] < sr:
        return None
    return measure_lufs(speech, sr)


def _check(name: str, measured, requirement: str, passed: bool | None,
           hint: str = "", advisory: bool = False) -> dict:
    return {"name": name, "measured": measured, "requirement": requirement,
            "passed": passed, "advisory": advisory, **({"hint": hint} if hint else {})}


def check(metrics: dict, spec_id: str, dialogue_lufs: float | None = None,
          file_info: dict | None = None) -> dict:
    """Pass/fail-rapport van metrics tegen één spec. dialogue_lufs is vereist
    voor dialogue-gated specs (via dialogue_loudness); file_info (io.probe)
    maakt de leveringsformaat-checks mogelijk voor specs die een formaat
    voorschrijven (Netflix: 48 kHz / 24-bit PCM)."""
    spec = SPECS.get(spec_id)
    if spec is None:
        raise ValueError(f"Onbekende spec '{spec_id}'. "
                         f"Beschikbaar: {', '.join(sorted(SPECS))}")
    checks: list[dict] = []

    fmt = spec.get("format")
    if fmt:
        sr_req = fmt.get("sample_rate")
        sr_meas = (file_info or {}).get("sample_rate") or metrics.get("sample_rate")
        if sr_req:
            checks.append(_check("Sample rate", sr_meas, f"{sr_req} Hz",
                                 sr_meas == sr_req,
                                 hint="" if sr_meas == sr_req else
                                 f"master_for(..., sample_rate={sr_req}) levert "
                                 "met sample-rate-conversie"))
        if fmt.get("min_bit_depth"):
            bits = (file_info or {}).get("bit_depth")
            subtype = (file_info or {}).get("subtype")
            pcm_ok = bool(subtype) and (subtype.startswith("PCM")
                                        or subtype in ("FLOAT", "DOUBLE"))
            if bits is None:
                checks.append(_check(
                    "Leveringsformaat", (file_info or {}).get("codec") or "onbekend",
                    f"PCM wav, >= {fmt['min_bit_depth']}-bit",
                    False,
                    hint="lossy/onbekend bronformaat: exporteer als wav via "
                         f"master_for(..., bit_depth={fmt['min_bit_depth']})"))
            else:
                ok = bits >= fmt["min_bit_depth"] and (pcm_ok or not fmt.get("pcm"))
                checks.append(_check(
                    "Leveringsformaat", f"{subtype} ({bits}-bit)",
                    f"PCM wav, >= {fmt['min_bit_depth']}-bit", ok,
                    hint="" if ok else
                    f"master_for(..., bit_depth={fmt['min_bit_depth']})"))

    loud = spec.get("loudness")
    if loud:
        if spec["gating"] == "dialogue":
            measured = dialogue_lufs
            label = "Dialogue-gated loudness"
            miss_hint = ("geen spraak gedetecteerd om op te meten; is dit wel "
                         "dialoogmateriaal?")
        else:
            measured = metrics.get("lufs_integrated")
            label = "Integrated loudness"
            miss_hint = "bestand te kort/stil voor een loudness-meting"
        measured = round(measured, 2) if measured is not None else None
        lo = loud["target"] - loud["tol"]
        hi = loud["target"] + loud["tol"]
        ok = measured is not None and lo <= measured <= hi
        checks.append(_check(
            label, measured, f"{loud['target']} LUFS ±{loud['tol']}",
            ok if measured is not None else False,
            hint=(miss_hint if measured is None else
                  "" if ok else f"master_for(spec='{spec_id}') corrigeert dit")))

    tp_max = spec.get("true_peak_max")
    if tp_max is not None:
        tp = metrics.get("true_peak_dbtp")
        checks.append(_check("True peak", tp, f"<= {tp_max} dBTP",
                             tp is not None and tp <= tp_max,
                             hint="" if (tp is not None and tp <= tp_max)
                             else "true-peak-limiter nodig (master_for)"))

    if "rms_range" in spec:
        lo, hi = spec["rms_range"]
        rms = metrics.get("rms_db")
        checks.append(_check("RMS-niveau", rms, f"{lo} tot {hi} dB",
                             rms is not None and lo <= rms <= hi))
    if "sample_peak_max" in spec:
        pk = metrics.get("sample_peak_db")
        checks.append(_check("Sample peak", pk, f"<= {spec['sample_peak_max']} dB",
                             pk is not None and pk <= spec["sample_peak_max"]))
    if "noise_floor_max" in spec:
        nf = metrics.get("noise_floor_db")
        checks.append(_check("Ruisvloer", nf, f"<= {spec['noise_floor_max']} dB",
                             nf is not None and nf <= spec["noise_floor_max"],
                             hint="" if (nf is not None and nf <= spec["noise_floor_max"])
                             else "reduce_noise of smart_edit"))

    if "lra_max_advisory" in spec:
        lra = metrics.get("lra_db")
        checks.append(_check("Loudness range", lra,
                             f"<= {spec['lra_max_advisory']} LU (richtlijn)",
                             lra is None or lra <= spec["lra_max_advisory"],
                             advisory=True))

    # universele technische QC-poorten
    checks.append(_check("Clipping", metrics.get("clip_events", 0), "0 clip-momenten",
                         metrics.get("clip_events", 0) == 0,
                         hint="" if metrics.get("clip_events", 0) == 0
                         else "repair_audio (declip)"))
    drops = (metrics.get("dropouts") or {}).get("count", 0)
    checks.append(_check("Dropouts", drops, "0 digitale gaten", drops == 0))
    stereo = metrics.get("stereo") or {}
    if stereo:
        checks.append(_check("Kanalen", stereo.get("dead_channel") or "beide actief",
                             "geen dood kanaal", stereo.get("dead_channel") is None))
        checks.append(_check("Polariteit",
                             stereo.get("correlation"),
                             "kanalen niet in tegenfase",
                             not stereo.get("polarity_inverted")))
    for edge, key in (("kop", "lead_silence_s"), ("staart", "tail_silence_s")):
        v = metrics.get(key)
        if v is not None:
            checks.append(_check(f"Stilte aan de {edge}", v, "<= 1 s (richtlijn)",
                                 v <= 1.0, advisory=True,
                                 hint="" if v <= 1.0 else "wegknippen voor aflevering"))

    normative = [c for c in checks if not c["advisory"]]
    passed = all(c["passed"] for c in normative)
    return {"spec": spec_id, "spec_name": spec["name"], "passed": passed,
            "failed_checks": [c["name"] for c in normative if not c["passed"]],
            "checks": checks}
