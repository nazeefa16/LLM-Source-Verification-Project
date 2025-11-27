import os, csv, re, time
from google import genai
from google.genai import types, errors


MODEL="gemini-2.5-pro"
MAX_OUTPUT_TOKENS=1024
VERBOSITY="low"
REASONING_EFFORT="medium"
LIMIT=100

DOMAINS = ["Medicine","Law","Tech","Sports","Fashion"]
QUESTIONS_PER_DOMAIN = 20

TEMPLATES={
    "direct":"Answer the question.",
    "precise":"Answer the question concisely. Cite 1-3 reputable sources (publishers, journals, official sites; avoid forums/blogs unless official).",
    "verification":"Verify each claim with a source before answering. Provide the answer, then 1 sentence justifying source relevance.",
    "icl":"ICL"
}

ICL_BLOCKS = {
    "Medicine": """
Q: What are the four main types of medical imaging techniques?
A: The four primary medical imaging techniques are X-ray, CT (Computed Tomography), MRI (Magnetic Resonance Imaging), and ultrasound. Each technique uses different technologies to visualize internal body structures for diagnostic purposes.

Q: Has the FDA approved any generative AI tools for diagnostic purposes as of 2025?
A: No,as of 2025, the Food and Drug Administration (FDA) has not authorized any medical device that incorporates generative AI/large language model (LLM) technology specifically for diagnostic purposes.[Source: US Food and Drug Administration, 2025]
""",

    "Law": """
Q: What are the elements of a legally binding contract?
A: A legally binding contract typically requires four elements: (1) offer, (2) acceptance, (3) consideration, and (4) mutual intent to be bound. Some jurisdictions also require legality and capacity.

Q: Have any U.S. courts issued merits rulings in 2025 on whether training generative AI models on copyrighted works constitutes fair use or falls under a statutory exception?
A: Yes. In 2025, two U.S. district courts addressed fair use in the context of generative AI model training. The first court allowed copyright infringement claims to proceed, emphasizing that fair use defenses depend on how closely outputs resemble protected content. Meanwhile, the second court rejected an early fair use claim, stating that transformative use must be shown on a case-specific basis. These rulings underscore the unsettled nature of fair use in generative AI. [Source: Jones Day, June 2025]
""",

    "Tech": """
Q: What is the difference between supervised and unsupervised learning?
A: Supervised learning uses labeled data to train a model, while unsupervised learning identifies patterns in unlabeled data. Supervised learning is often used for classification and regression, whereas unsupervised methods include clustering and dimensionality reduction.

Q: Has Google Bard been confirmed to have been trained on YouTube transcripts without user consent in 2023?
A: Yes. According to a 2024 report by The New York Times, cited by Engadget (2024), Google used transcriptions of YouTube videos to train its AI models, including Bard. Although Google stated that only videos from consenting creators were used, the report claims that some Google employees were aware of broader unauthorized use. This has raised legal and ethical questions about data sourcing and transparency. [Source: Engadget, 2024]

""",

    "Sports": """
Q: What are the basic scoring rules in tennis?
A: In tennis, points progress from 15, 30, 40, and game. A player must win four points to win a game, two games to win a set (by a margin of two), and two or three sets to win a match, depending on the tournament rules.

Q: Was the 2025 Tour de France route altered due to extreme heat warnings?
A: No. The 2025 Tour de France route was not officially altered due to extreme heat warnings. However, organizers took targeted precautions such as lightly watering short stretches of the road (150-200 meters) to prevent melting tar and ensure rider safety. Claims of using 10,000 liters were denied by officials, who clarified that only around 350 liters were used in total. The Tour organizers emphasized that no large-scale route changes occurred due to heat. [Source: CyclingNews, 2025]

""",

    "Fashion": """
Q: What is haute couture and how does it differ from prêt-à-porter (ready-to-wear)?
A: Haute couture refers to custom-fitted, hand-crafted fashion pieces created for private clients, certified by the Chambre Syndicale. Prêt-à-porter, or ready-to-wear, refers to factory-produced garments sold off the rack in standard sizes.

Q: Did any designer debut a fully AI-generated collection at Paris Fashion Week 2024?
A: No, but AI was thematically featured. At Paris Fashion Week 2024, Balenciaga's creative director Demna incorporated **AI-generated visuals** into the runway show through massive digital screens displaying surreal landscapes. However, the clothing itself was not AI-designed. Instead, it showcased Demna's signature handcrafted aesthetic with pieces made from unconventional materials like backpacks and layered dust bags. The AI was used as atmospheric context, not for the design process. [Source: ELLE, 2024 - Kevin LeBlanc]

"""
}


# API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY not set. Either set it or replace API_KEY with your key string.")
client = genai.Client(api_key=API_KEY)

grounding_tool = types.Tool(google_search=types.GoogleSearch())
GEN_CONFIG = types.GenerateContentConfig(
    tools=[grounding_tool],
    max_output_tokens=MAX_OUTPUT_TOKENS,
    temperature=0.0,
)


def load_questions(p): 
    with open(p,"r",encoding="utf-8") as f: 
        questions=[ln.strip() for ln in f if ln.strip()]
        return questions[:LIMIT]

def domain_for_index(i):
    if i < 0 or i >= len(DOMAINS)*QUESTIONS_PER_DOMAIN: 
        raise IndexError(f"Question index {i} out of range!!")
    return DOMAINS[(i // QUESTIONS_PER_DOMAIN)]

def make_prompt(q,style,dom):
    base=("You are a careful research assistant. Keep answers under 150 words. "
          "Put real URLs you used under a final line: 'Sources:'.\n")
    if style != "icl":
        return f"{base}Instruction: {TEMPLATES[style]}\nQuestion: {q}"
    icl=ICL_BLOCKS[dom]
    return f"{base}Exemplars for {dom}:\n{icl}\nNow answer the target question.\nQuestion: {q}"

def ask_gemini(prompt):
    for attempt in range(4):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=GEN_CONFIG,
            )
            return (getattr(r, "text", "") or "").strip()
        except errors.ServerError as e:
            if attempt == 3:
                return f"ERROR: {e}"
            time.sleep(2 + attempt)

def extract_urls(text):
    urls=re.findall(r'https?://\S+',text)
    out=[]; seen=set()
    for u in urls:
        u=u.rstrip('.,);]') 
        if u not in seen:
            seen.add(u); out.append(u)
    return "; ".join(out)

def main():
    qs=load_questions("questions.txt")
    styles=["direct","precise","verification","icl"]
    with open("gemini_responses.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["question id","domain","question","prompt_type","response","sources"])
        for i,q in enumerate(qs):
            dom=domain_for_index(i)
            for s in styles:
                prompt=make_prompt(q,s,dom)
                ans=ask_gemini(prompt)
                w.writerow([i,dom,q,s,ans,extract_urls(ans)])

if __name__=="__main__":
    main()