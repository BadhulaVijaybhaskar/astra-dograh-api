import asyncio, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from api.services.pipecat.tts_phrase_aggregator import NaturalPhraseAggregator, LegacyPhraseTextAggregator

async def run(agg, text, step=3):
    out = []
    for i in range(0, len(text), step):
        async for a in agg.aggregate(text[i:i+step]):
            out.append(a.text)
    f = await agg.flush()
    if f and f.text: out.append(f.text)
    return out

CASES = {
 "brs": "BRS టీమ్ నుండి మాట్లాడుతున్నాను.",
 "q": "మీ area లో main problem ఏంటి?",
 "turn1": "రామ్ గారు, KTR మాట్లాడుతున్నాను, BRS టీమ్ నుండి. ఈ సారి మాట్లాడటానికి సమయం ఉందా?",
 "close": "సరే, రామ్ గారు. మీ అభిప్రాయం చెప్పినందుకు ధన్యవాదాలు. ఇంకేమైనా సమస్యలు గుర్తొచ్చినప్పుడు ఈ నంబర్‌కి మెసేజ్ చేయండి.",
 "ellipsis": "అవును… drainage మరియు water problems, చాలా ముఖ్యమైనవి. సరే... ఇంకా ఏమైనా?",
 "num": "Plan ధర Rs 14,999 మాత్రమే, 2.5 నెలలు free support ఉంటుంది. మీకు ఆసక్తి ఉందా?",
}
async def main():
    fails = 0
    for k, t in CASES.items():
        new = await run(NaturalPhraseAggregator(), t)
        old = await run(LegacyPhraseTextAggregator(), t)
        print(f"[{k}]\n  old: {old}\n  new: {new}")
        assert "".join(new).replace(" ", "") == t.replace(" ", ""), k  # text preserved
        for p in new[:-1]:
            if len(p) < 25: print("  FAIL tiny", p); fails += 1
        for p in new:
            if p.endswith("…") or p.endswith("..."): print("  FAIL ellipsis split", p); fails += 1
    assert await run(NaturalPhraseAggregator(), CASES["brs"]) == [CASES["brs"]]
    assert await run(NaturalPhraseAggregator(), CASES["q"]) == [CASES["q"]]
    n = await run(NaturalPhraseAggregator(), CASES["turn1"])
    assert n == [CASES["turn1"].split(". ")[0] + ".", "ఈ సారి మాట్లాడటానికి సమయం ఉందా?"], n
    n = await run(NaturalPhraseAggregator(), CASES["num"])
    assert all("14,999" in p or "14" not in p for p in n) and not any(p.endswith("2.") for p in n), n
    print("FAILS", fails); assert fails == 0
asyncio.run(main())
