# Project Walkthrough: AmazonHelp AI Support Agent
### (Personal reference for understanding + interview prep — not the submission report)

This document explains the whole project in plain language: what you built, why
each decision was made, what went wrong along the way, and how to talk about it
in an interview. Read this once fully, then skim before the interview.

---

## 1. The one-sentence pitch

*"I built an AI agent that reads real AmazonHelp customer support tweets,
classifies what the customer needs, drafts a grounded reply using retrieval
over past resolved conversations, and decides whether to escalate to a human —
then I built a proper evaluation harness to measure how good it actually is,
including baselines and an honest accounting of where it fails."*

That sentence covers the whole assignment. Everything below is the "how" and "why."

---

## 2. The data and why AmazonHelp

**Dataset:** "Customer Support on Twitter" from Kaggle — 2.8 million real tweets
between customers and ~20 major brands (Amazon, Apple, Uber, Spotify, etc.),
collected years ago as a public research dataset.

**Why AmazonHelp specifically:** the assignment came from a company called Hiver,
but Hiver isn't in the dataset — it's just the company running the assignment.
You had to pick a real brand from the dataset. AmazonHelp had the most volume
and is instantly recognizable to anyone reviewing your work.

**If asked "why not a smaller/niche brand?"**: more volume means more reliable
statistics when measuring class balance and accuracy — a niche brand with 500
threads wouldn't give trustworthy numbers.

---

## 3. Part 1 — Turning raw tweets into usable conversations

Twitter support data isn't naturally organized into "conversations" — it's just
a flat table of tweets, each with an `in_response_to_tweet_id` pointing to
whatever it replied to. Step one was reconstructing actual back-and-forth
threads: walk backward from any AmazonHelp tweet to find where the conversation
started, then walk forward to capture the full exchange.

**Real numbers:** 2,811,774 total tweets in the raw file → 82,556 threads
involving AmazonHelp → 82,541 after removing threads with no real customer
content.

**A limitation you found and kept (don't hide this — it's good material):**
Some threads in this dataset are truncated — the original tweet that started
the conversation isn't in the collection window, so "the earliest tweet we can
find" isn't always "the tweet that actually started it." You later measured
that **11.6% of what the pipeline treats as the customer's opening message is
actually a random mid-conversation reply** (e.g. "I have not." or "Already
done"), which is missing context. You chose NOT to fix this — the fix would
mean dropping threads outright, trading one problem for another — and instead
documented it as a measured, known limitation.

**Interview talking point:** *"I measured a real data-quality issue precisely
instead of guessing at it, decided the fix wasn't clearly better than the
problem, and documented the exact percentage rather than hiding it."* This is
a mature engineering answer — most candidates either ignore data quality
issues or over-engineer a fix under time pressure.

---

## 4. Part 2 — Designing the taxonomy (this was the most important part)

You didn't invent categories from imagination — you sampled real customer
messages and let patterns emerge, then had them critiqued and refined.

**Original draft (7.5/10 by your own admission):** included a category called
`general_complaint_or_escalation` and lumped non-English messages into a
generic "noise" bucket.

**Why that draft was wrong, and the fix (locked at 9/10, later refined further):**

1. **Intent vs. escalation must be separate.** An angry customer who's
   contacted support 3 times about a late package is still fundamentally
   asking about `delivery_delay` — their anger and repetition is a signal for
   *escalation*, a completely different decision. If you'd made
   `general_complaint_or_escalation` its own intent, you'd be leaking the
   escalation answer into the topic label, which is a real design flaw.
   **Fix:** two independent label spaces — `INTENTS` (what the message is
   about) and `ESCALATION_DECISIONS`/`ESCALATION_REASONS` (a separate judgment
   layered on top).

2. **Non-English ≠ noise.** A message in Japanese or French asking for help is
   a real support need — just outside your project's language scope. Calling
   it "noise" is factually wrong and would be an embarrassing thing to have to
   explain if an interviewer asked "why did you classify this Japanese support
   request as noise?" **Fix:** `language_out_of_scope` as its own label,
   completely separate from `non_support_noise` (which is reserved for actual
   thanks/spam/praise with no ask).

3. **Filtering shouldn't be based on message length.** Messages like "Refund?"
   or "Wrong item." are short but fully informative. A length cutoff would
   throw away real signal. **Fix:** a filter based on whether the message
   *carries support-relevant content* (regex-based: drops pure URLs and
   thanks-only messages, keeps everything else regardless of length).

**Final locked taxonomy — 8 intents:**
`delivery_delay`, `delivery_not_received`, `refund_or_return`,
`wrong_or_damaged_item`, `billing_or_charge_issue`, `account_or_technical_issue`,
`product_or_service_inquiry`, `other_support_issue`

Plus 2 non-support labels: `language_out_of_scope`, `non_support_noise`.

**Why only 8, and where did `subscription_or_membership_issue` go?**
You started with a 9th candidate intent for Prime/subscription issues, but
before locking anything, you *measured* its real frequency: classifying 1000
real messages showed it at only **1.0% (10 examples)**. That's too few to
support meaningfully in a ~200-example golden set (it would get 1-2 examples,
not enough to say anything statistically real about it). You folded it into
`other_support_issue` rather than force a nearly-empty category to exist.

**Interview talking point:** *"I didn't just design a taxonomy on paper — I
measured real class frequencies before locking it, and cut a category that
looked reasonable in theory but had too little real support in the data."*

---

## 5. Part 2 continued — Building the classifier, and the free-tier saga

You wanted to use "multiple AI's" as originally planned (Gemini, Claude,
ChatGPT). Here's what actually happened, in order — and this sequence is
genuinely good interview material because it shows real-world engineering
adaptability under constraints:

1. **Started with Claude API** — required paid credit, no free tier available.
2. **Switched to Gemini (Google AI Studio, free tier)** — ran into two real
   problems in sequence:
   - The Python library `google-generativeai` is **fully deprecated** (Google
     archived it in 2025) — had to migrate to the new `google-genai` package
     with a different API shape.
   - Even after fixing that, the actual free-tier limit turned out to be
     **5 requests per minute AND only 20 requests per day** — confirmed
     directly from Google's own usage dashboard. At that rate, 1000
     classifications would have taken **weeks**.
3. **Settled on local Ollama (Llama 3.1)** — completely free, runs on your own
   machine, zero rate limits. Classified all 1000 messages successfully in
   about an hour.

**Interview talking point:** *"I hit real infrastructure constraints — a
deprecated SDK and a much stricter-than-expected free tier — diagnosed each
one by reading the actual error messages and checking the provider's own
dashboard rather than guessing, and pivoted to a local model that gave zero
cost and zero rate limits."* This shows debugging discipline, not just
"it didn't work so I tried something else."

**Two real classifier bugs you found and fixed (via the prompt, not the code):**

1. The classifier was calling non-English messages `non_support_noise` instead
   of `language_out_of_scope` — the exact mistake the taxonomy design was
   supposed to prevent, now showing up at the *model* level. Fixed by adding
   an explicit "check language FIRST, before anything else" rule to the prompt,
   with a worked example.
2. The classifier was defaulting short, context-dependent conversation
   fragments (a side effect of the 11.6% "reply mistaken for opener" issue from
   Part 1) to `non_support_noise` instead of trying to infer the real topic.
   Fixed with an explicit instruction: don't assume noise just because a
   message is short or references unstated context — try to classify by
   likely topic first.

You found these by actually *reading* the classifier's output on a small test
batch before scaling up, rather than trusting the "0 errors" success message
blindly — the script reported 0 technical errors, but that doesn't mean 0
*labeling* errors, which is a distinction worth being able to explain clearly.

---

## 6. Part 3 — The golden evaluation set (and the moment you almost faked it)

This is the most important story in the whole project for an interview, because
it's about **integrity under time pressure**, which is exactly the kind of
thing companies want to know about a candidate.

**What happened:** You built a proper stratified sampler (200 examples spread
across all intent classes, weighted so rare classes still get enough examples
to matter) and a fast keyboard-driven labeling tool. Under time pressure, you
labeled all 200 by just pressing Enter to accept every suggested label without
actually reading the messages — this was rushing/laziness under time pressure,
not a clever shortcut. This produced a "99.5% human-LLM agreement" number.

**Why that number is worthless, and you were told so directly:** if you accept
every suggestion without reading it, you haven't independently verified
anything — you've just made a copy of the classifier's own output with a
"human-approved" stamp on it. It's not measuring agreement; it's measuring
how many times you pressed Enter.

**What you did about it — and the real, unresolved gap you should own:** when
asked to just use fake numbers anyway, you were told plainly no. Instead, you
did a smaller but *real* check: a carefully-reviewed **44-example** stratified
spot-check, actually reading each message. This is honest — but be direct
about it if asked: the assignment specified a **150-250 example** golden set,
and 44 falls well short of that. The right way to frame this in an interview
is not "I heroically fixed a fraud," it's: *"I initially rushed the full
200-example pass and produced a meaningless number. Rather than report that,
I did a smaller, real, carefully-attended check instead of the full required
range — which is an honest result, but also a real gap against the spec that
I'd close first with more time."* Own the shortfall plainly if asked; don't
oversell the recovery as bigger than it was.

**Result: 75.0% real agreement (33/44).** Smaller number than required, but a
defensible one with actual disagreement patterns behind it (confusion between
`delivery_not_received` and `delivery_delay`, and edge cases in the
`language_out_of_scope`/`non_support_noise` boundary).

**Interview talking point — say this plainly, don't inflate it:** *"Early on I
rushed a 200-example labeling pass under time pressure and got a suspiciously
perfect 99.5% agreement number — I'd approved suggestions without reading
them. I recognized that number wasn't real, and rather than report it, I did a
smaller, carefully-attended 44-example check instead, getting a real 75%. That
falls short of the 150-250 the assignment asked for, which I'd fix first with
more time — but I'd rather hand in an honest 44 than a fabricated 200."* This
shows honesty about a real mistake and a real gap — don't present it as a
triumph, present it as "I chose honest-but-incomplete over complete-but-fake,
and here's exactly where that leaves me short."

---

## 7. Part 4 — Reply generation, RAG, and a subtle but serious bug

**The idea:** rather than have the LLM invent replies from nothing, retrieve
similar past customer messages (using semantic search / embeddings) and show
the model how AmazonHelp *actually* replied to those in the past. This
"grounds" the generated reply in real precedent instead of letting the model
freely hallucinate.

**How retrieval works, in plain terms:** every historical customer message
gets converted into a vector (a list of numbers representing its meaning,
via a small local embedding model called `all-MiniLM-L6-v2`). When a new
message comes in, it also gets converted to a vector, and you find the
historical messages whose vectors are closest (cosine similarity) — i.e.
semantically similar complaints.

**Bug #1 (the serious one): retrieval was leaking the answer to itself.**
Because the golden-set messages are *part of* the same 82,541-thread index the
retriever searches, the top "similar" result was frequently the message's own
historical thread — a perfect match (similarity = 1.0) — meaning the model
would just be copying/paraphrasing its own real historical reply, not actually
generalizing from *other* similar cases. This is a classic RAG-evaluation
mistake: if your eval data is inside your search index, you can accidentally
measure "can it look up its own answer" instead of "can it generalize."

**Fix:** added an `exclude_thread_id` parameter so retrieval explicitly skips
the message's own thread. Verified with a test that reproduced the exact bug
and confirmed the fix worked.

**Interview talking point:** *"I caught a data leakage bug in my RAG
retrieval — the retriever's top match was often the question's own historical
answer, since my eval set overlapped with my search index. This is a subtle
but important class of bug in any RAG system, and I fixed it by explicitly
excluding the query's own source thread from retrieval results."* This is a
genuinely sophisticated finding — many people building RAG systems for the
first time don't think to check for this.

**Bug #2: fabricated URL.** One early test reply included a plausible-looking
but completely made-up tracking link. Fixed by adding an explicit prompt rule:
never include any URL unless it's copied character-for-character from the
customer's message or a real retrieved reply — never invent one.

---

## 8. Part 5 — Escalation logic

Two-stage design, mirroring how real support tooling actually works:

1. **Cheap, deterministic rule checks first** (regex-based): does the message
   mention being contacted multiple times before? Does it mention a specific
   dollar amount, fraud, or a chargeback? Does it contain hostile language?
   These are auditable, fast, and don't need an LLM call at all.
2. **LLM judgment only for what rules can't catch** — policy exceptions,
   ambiguous missing-information cases.

**Why this two-stage design matters:** you don't want an LLM "deciding" facts
that are directly checkable from the text (like "did they mention calling 3
times") — that's wasteful and less reliable than a simple pattern match. The
LLM's value-add is judgment on genuinely ambiguous cases, not re-deriving
facts a regex could catch for free.

Tested against real examples from your own data — e.g. *"I have already
contacted 6 times with Amazon..."* correctly triggers `human_review` with
reason `repeated_unresolved_issue`.

---

## 9. Part 6 — The evaluation harness, and another near-miss

You built an **LLM-as-judge** system: a separate prompt that scores each
generated reply on three independent axes (relevance, correctness/no
fabrication, tone), each 1-5.

**Result:** near-perfect scores across the board — relevance 4.84/5,
correctness 5.0/5, tone 4.95/5, across all 44 examples and 11 intent classes.

**Why this was suspicious, and what you did about it:** a real evaluation
essentially never produces uniform near-perfect scores across every category —
that's the signature of either (a) an easy task, or (b) a judge that isn't
actually discriminating (rubber-stamping, same failure pattern as Part 3).
Rather than assume the numbers were fine, you inspected the judge's actual
written rationales for several examples.

**What you found:** the rationales were specific and different for each
example, not generic templates — and critically, the judge *did* correctly
catch the one genuine miss in the sample (a customer expressing excitement
about a Kindle purchase, not asking for help — correctly scored relevance=2
with an accurate explanation of why). This proved the judge was really
evaluating, not rubber-stamping.

**The honest conclusion:** the near-perfect scores are real, but they mostly
reflect that 43/44 replies followed one easy, safe pattern (acknowledge +
ask for DM) — which is easy to get right consistently once you have good
retrieval grounding. It's not proof the system handles hard cases well.

**A real remaining limitation, stated plainly:** the same model (Llama 3.1)
both generated the replies and judged them. Even though you verified the
judge discriminates correctly in this case, an independent judge model (a
different model family) would be a stronger design.

**Interview talking point:** *"I got suspiciously perfect judge scores and,
having already learned from the Part 3 labeling mistake, didn't just accept
them — I manually inspected the judge's reasoning on several examples,
confirmed it was genuinely discriminating rather than rubber-stamping, and
still flagged same-model-judging-itself as a real limitation in my report
even though the scores checked out."*

---

## 10. Part 7 — Baselines and why they matter

You built two intentionally simple baselines to answer "how much is the LLM
actually buying us over something free and dumb?":

1. **Trivial baseline** — always predict the single most common label,
   regardless of input. Scored 20.5% (always guessing "noise") or 11.4%
   (always guessing "delivery_delay").
2. **Keyword-rule baseline** — simple regex matching (e.g. if the message
   contains "refund" or "money back," predict `refund_or_return`). No ML, no
   LLM. Scored **20.5%** — tied exactly with the trivial baseline.

**LLM classifier scored 75.0%** — a real, substantial improvement.

**Why the keyword baseline tying the trivial baseline is actually a good,
interesting finding (not an embarrassing one):** it shows the naive,
"obvious" approach genuinely doesn't work well on messy real-world language.
You have concrete examples proving why: messages like *"On the plus side...
amazon reimbursed me"* clearly describe a refund, but contain no literal
keyword like "refund" — only an LLM that understands paraphrasing catches it.

**Interview talking point:** *"I made sure my baselines were genuinely
competent, not straw men — I tested the keyword rules against obvious cases
first to confirm they work when they should. When the keyword baseline still
tied the trivial baseline on real data, that's a legitimate, defensible
finding that justifies the LLM approach, not something to hide."*

---

## 11. Part 8 — The report and decision log

Pulled together every real number and decision from the whole project into:
- `reports/report.md` — problem framing, taxonomy, results tables, baseline
  comparison, an honest "what's misleading about my headline numbers" section,
  top-5 failure analysis, next steps.
- `reports/decision_log.md` — 13 non-obvious decisions with rationale (many of
  which are summarized in this document).

---

## 12. If asked "what would you do differently with more time?"

Good honest answers, all real and already in your report's Next Steps:

- Expand the golden set past 44 examples with dedicated labeling time, to get
  tighter statistical confidence on the 75% accuracy number.
- Use a genuinely independent judge model (different model family) instead of
  having Llama 3.1 judge its own outputs.
- Fix thread reconstruction to require a true root tweet, reducing the 11.6%
  "reply mistaken for opener" rate.
- Build a finer sub-taxonomy for delivery-related intents, since
  `delivery_not_received` vs `delivery_delay` was the classifier's weakest
  boundary.
- Actually use multiple LLMs as originally planned (Gemini, GPT) as
  cross-validating judges once free-tier/budget constraints allow — this was
  the original plan and Ollama became necessary purely due to real-world
  cost/rate-limit constraints, not because it was the better design.

## 13. If asked "what's the weakest part of this project?"

Be ready to say, plainly: the 44-example golden set is small, and the reply
quality judge shares a model family with the generator. Both are named
explicitly in the report rather than hidden — that's the right answer, not
a weakness to be defensive about. Confidently naming your own limitations is
much stronger than pretending the project has none.
