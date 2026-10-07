"""Serial three-run prose/code/math performance acceptance on the local GPU.

Reports the median of nine 256-token runs and whether it reaches 40 tokens/s.
Use --long for separate 4K/16K/32K/128K full-history retrieval profiles.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

PROMPTS = {
    "prose": "Write a detailed travel journal about a scientist visiting a remote mountain village. Describe the journey, the weather, the people she meets, the scientific observations she records, and the way the landscape changes throughout the day. Use concrete details and connected paragraphs. Include a small unexpected event and explain how she deals with it. Continue the narrative for at least six paragraphs, without headings or bullet points.",
    "code": "Explain binary search and implement a lower_bound function in Python. Include type annotations, a clear docstring, the loop invariant, time and space complexity, and worked examples involving an empty list, duplicate elements, a missing target, and a target larger than every element. Then show how to use the function to insert a value into a sorted list while preserving order. Give a detailed explanation that a beginner can follow.",
    "math": "Explain how to solve a quadratic equation using the quadratic formula. Derive the formula by completing the square, explain the discriminant and the three possible cases, and work through the examples x squared minus five x plus six equals zero, x squared plus four x plus four equals zero, and x squared plus one equals zero. Show every algebraic step, verify the real roots by substitution, and discuss the complex roots in the final example.",
}


def main():
    root = Path(__file__).resolve().parents[1]; data = root.parent / "Lamina-data"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, default=root / "build-cuda/lamina-infer.exe")
    p.add_argument("--report-dir", type=Path, default=data / "performance")
    p.add_argument("--compute-mode", choices=("f32","fast"), default="fast")
    p.add_argument("--prompts",nargs="+",choices=tuple(PROMPTS),default=list(PROMPTS))
    p.add_argument("--long", action="store_true")
    p.add_argument("--only-long", action="store_true")
    p.add_argument("--retrieval-filler", choices=("varied","repeat"), default="varied")
    p.add_argument("--contexts",type=int,nargs="+",default=[4096,16384,32768,131072],choices=(4096,16384,32768,131072))
    a=p.parse_args(); a.report_dir.mkdir(parents=True,exist_ok=True)
    def run(name,options):
        output=a.report_dir/(name+".json")
        cmd=[sys.executable,"-m","tools.benchmark","--engine",str(a.engine.resolve()),"--cuda","--compute-mode",a.compute_mode,"--quiet","--chunk","2048","--json",str(output.resolve()),*options]
        subprocess.run(cmd,check=True)
        return json.loads(output.read_text(encoding="utf-8"))
    reports=[]
    if not a.only_long:
        for name in a.prompts:
            prompt=PROMPTS[name]
            for repeat in range(3):
                reports.append(run(f"{name}-{repeat+1}",["--prompt",prompt,"--tokens","256"]))
        median=statistics.median(r["later_tokens_per_second"] for r in reports)
        summary={"median_later_tokens_per_second":median,"target":40,"acceptance_complete":set(a.prompts)==set(PROMPTS), "target_met":median>=40 and set(a.prompts)==set(PROMPTS),
                 "prompt_medians":{name:statistics.median(r["later_tokens_per_second"] for r in reports[i*3:i*3+3]) for i,name in enumerate(a.prompts)},
                 "reports":[f"{name}-{i}.json" for name in a.prompts for i in range(1,4)]}
        (a.report_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(summary),flush=True)
    if a.long or a.only_long:
        from tokenizers import Tokenizer
        from lamina import chat_prompt
        tokenizer=Tokenizer.from_file(str(data/"tokenizer/tokenizer.json"))
        passage="The field notebook records rainfall, soil temperature, plant growth and the time of every observation. Each measurement is checked carefully before it is entered into the table. The researchers compare successive days and retain the original records for later analysis.\n"
        needle="The unique vault code for station Amber is LAMINA847263.\n"
        question="\nWhat is the unique vault code for station Amber? Reply with the code alone."
        for context in a.contexts:
            target=context-128
            # Place the answer near the beginning, beyond any short sliding window.
            prefix=passage*4+needle
            if a.retrieval_filler=="repeat":
                passages=[passage]*(target//len(tokenizer.encode(passage,add_special_tokens=False).ids))
            else:
                passages=[f"Field record {day}: the northern plot received {day*17%83} millimetres of rain. The soil temperature was {day*13%29+3} degrees. Researchers counted {day*19%997+20} seedlings in row {day%31+1}, inspected the measuring instruments and recorded the observations before leaving the station.\n" for day in range(1,target//32+1)]
            low,high=0,len(passages)
            while low<high:
                middle=(low+high+1)//2
                candidate=prefix+"".join(passages[:middle])+question
                count=len(tokenizer.encode(chat_prompt([{"role":"user","content":candidate}]),add_special_tokens=False).ids)
                if count<=target:low=middle
                else:high=middle-1
            text=prefix+"".join(passages[:low])+question
            path=data/f"retrieval-{a.retrieval_filler}-{context}.txt";path.write_text(text,encoding="utf-8")
            options=["--prompt-file",str(path.resolve()),"--max-context",str(context),"--tokens","64", "--stop-at-eos", "--progress-file",str((data/f"retrieval-{context}-progress.json").resolve())]
            if context==131072: options += ["--kv-type","f16","--kv-cache","device"]
            report=run(f"retrieval-{context}",options)
            answer=tokenizer.decode(report["next_tokens"],skip_special_tokens=True)
            report["retrieval_filler"]=a.retrieval_filler;report["retrieval_answer"]=answer;report["expected_answer"]="LAMINA847263";report["retrieval_passed"]="LAMINA847263" in answer
            (a.report_dir/f"retrieval-{context}.json").write_text(json.dumps(report,indent=2,default=str)+"\n",encoding="utf-8")
            print(context,"retrieval",report["retrieval_passed"],repr(answer),flush=True)


if __name__=="__main__": main()
