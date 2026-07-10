"""Dev-CLI: snel analyseren/verbeteren zonder MCP. `uv run ait --help`."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="ait", description="Audio Improve Toolkit dev-CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="analyseer een audiobestand")
    pa.add_argument("file")
    pa.add_argument("--json", action="store_true", help="volledige JSON-uitvoer")

    pi = sub.add_parser("improve", help="auto-improve een audiobestand")
    pi.add_argument("file")
    pi.add_argument("--profile", default="auto", choices=["auto", "speech", "music"])
    pi.add_argument("--target-lufs", type=float, default=None)
    pi.add_argument("--denoise-method", default="auto", choices=["auto", "spectral", "ai"])
    pi.add_argument("--out", default=None, help="exportpad (formaat volgt de extensie)")

    pr = sub.add_parser("refine", help="iteratief verfijnen tot doelen kloppen")
    pr.add_argument("file")
    pr.add_argument("--speech-peak", type=float, default=-6.0)
    pr.add_argument("--gap", type=float, default=2.0, help="muziek t.o.v. spraak (dB)")
    pr.add_argument("--iterations", type=int, default=5)
    pr.add_argument("--denoise", default="auto", choices=["auto", "on", "off"])
    pr.add_argument("--out", default=None)

    pv = sub.add_parser("viewer", help="start de A/B-viewer")
    pv.add_argument("--port", type=int, default=None)

    args = p.parse_args()

    if args.cmd == "analyze":
        from audio_improve_toolkit import analysis, io

        x, sr = io.load_audio(args.file)
        m = analysis.analyze(x, sr)
        scores, issues = analysis.score_and_issues(m)
        if args.json:
            print(json.dumps({"metrics": m, "scores": scores, "issues": issues},
                             indent=2, ensure_ascii=False))
        else:
            print(f"== {args.file} ==")
            for k, v in m.items():
                print(f"  {k}: {v}")
            print("scores:", scores)
            for i in issues:
                print(f"  [{i['severity']}] {i['message']} -> {i['suggestion']}")

    elif args.cmd == "improve":
        from audio_improve_toolkit import analysis, chain, improve, io, sessions

        x, sr = io.load_audio(args.file)
        m0 = analysis.analyze(x, sr)
        profile, steps, rationale = improve.build_improve_chain(
            m0, profile=args.profile, target_lufs=args.target_lufs,
            denoise_method=args.denoise_method)
        y, resolved = chain.run_chain(x, sr, steps)
        m1 = analysis.analyze(y, sr)
        session = sessions.create_session(args.file, x, sr, m0, y, m1, resolved,
                                          rationale, profile)
        if args.out:
            wav = sessions.session_path(session["session_id"]) / "processed.wav"
            out = io.encode_wav_to(wav, args.out)
            print(f"export: {out}")
        print(f"sessie: {session['session_id']}")
        for r in rationale:
            print(f"  - {r}")
        print("deltas:", json.dumps(session["deltas"], ensure_ascii=False))

    elif args.cmd == "refine":
        from audio_improve_toolkit import server

        res = server.refine_audio(args.file, speech_peak_db=args.speech_peak,
                                  music_gap_db=args.gap, max_iterations=args.iterations,
                                  denoise=args.denoise, out_path=args.out)
        print(f"sessie: {res['session_id']}")
        for r in res["rationale"]:
            print(f"  - {r}")
        print("eindmeting:", json.dumps(res["report"]["final_measurements"],
                                        ensure_ascii=False))
        if res["report"].get("asr"):
            print("whisper:", json.dumps({k: v for k, v in res["report"]["asr"].items()
                                          if not k.startswith("transcript")},
                                         ensure_ascii=False))

    elif args.cmd == "viewer":
        from audio_improve_toolkit.viewer.server import main as viewer_main

        viewer_main(port=args.port)


if __name__ == "__main__":
    main()
