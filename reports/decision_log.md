# Decision Log

Non-obvious decisions made during this project, with rationale. Ordered roughly
by when they came up.

**1. Target brand: AmazonHelp, not a brand named "Hiver"**
The assignment is from a company called Hiver, but the dataset itself only
contains real brand support accounts (AmazonHelp, AppleSupport, etc.). Chose
AmazonHelp for volume and recognizability.

**2. Separated intent classification from escalation decision-making**
Initially considered a `general_complaint_or_escalation` intent class. Rejected
this: it would leak the escalation answer into the intent label (an angry,
repeated complaint about a late package is still fundamentally about the
delivery, not a new topic). Escalation is modeled as an independent decision
layered on top of intent, with its own reason codes.

**3. Filtering rule: signal-based, not length-based**
Initially planned to filter out short messages (<15 chars) before
classification. Rejected in favor of a signal-based filter: "Refund?" and
"Wrong item." are short but fully informative; length alone is an arbitrary
and misleading proxy for whether a message is worth classifying.

**4. `language_out_of_scope` kept distinct from `non_support_noise`**
Non-English messages are not noise — many are genuine, urgent support requests
(one example in the raw data was an angry, non-English message about money
being stuck, clearly a real complaint). Conflating the two would mean silently
mislabeling real support needs as garbage.

**5. `subscription_or_membership_issue` folded into `other_support_issue`**
Measured at only 1.0% (10/1000) in the classified sample -- too thin to justify
as a standalone class in a ~200-example golden set, where it would get at most
1-2 examples. Folded in rather than force-split a near-empty category.

**6. Switched classification backend twice: Claude API -> Gemini free tier ->
local Ollama**
Claude API required paid credit; Gemini's free tier proved unusable at scale
(20 requests/day limit, confirmed via the AI Studio usage dashboard -- would
have taken weeks for 1000 classifications). Settled on local Ollama (Llama 3.1):
zero cost, no rate limits, fully reproducible without any API key.

**7. Chose not to re-engineer thread reconstruction after measuring an 11.6%
fragment-as-opener rate**
Diagnosed that "first customer turn" extraction sometimes grabs a mid-conversation
reply rather than a true thread opener (measured at 11.6% across all 82,541
threads). The fix (requiring a true root tweet, no `in_response_to_tweet_id`)
would silently drop threads rather than clearly improve accuracy. Documented as
a measured, known limitation instead of patching under time pressure.

**8. Rejected a fabricated 200-example "99.5% agreement" golden set, at the
cost of falling short of the 150-250 spec**
An initial labeling pass was rushed (rubber-stamped without reading messages).
Rather than report that number, redid it as an honest, carefully-reviewed
44-example stratified spot-check (75.0% real agreement). A smaller, trustworthy
number was judged more valuable than a larger, fake one -- but 44 is short of
the 150-250 examples this assignment specifies, and that gap is real, not
resolved by the honesty of the smaller check. Listed as the top item in
Next Steps.

**9. Reply generation explicitly forbidden from inventing URLs or specific
commitments**
Early test generations fabricated a plausible-looking but fake tracking URL.
Added an explicit prompt rule: never include a URL unless copied verbatim from
the customer's message or a retrieved real brand reply. Same principle applied
to refund amounts, dates, and compensation promises.

**10. Retrieval index excludes the query's own thread**
Found that the RAG retriever's top "similar" match was frequently the query
message's own historical thread (similarity=1.0) when the golden set overlaps
with the index it searches -- a direct data leak. Added an `exclude_thread_id`
parameter to prevent the generator from trivially "grounding" a reply by
looking up its own answer.

**11. LLM-as-judge scores were investigated before being trusted, not accepted
at face value**
Near-perfect reply-quality scores (4.84-5.0/5 across all 44 examples and 11
intent classes) were suspicious on their face -- the same failure pattern as
decision #8. Manually inspected judge rationales and confirmed they were
specific and discriminating (the judge correctly caught the one genuine miss
in the sample), concluding the scores reflect a genuinely easy sub-task
pattern rather than a broken or lenient judge. Documented same-model-judges-
itself as a residual limitation regardless.

**12. Two baselines chosen specifically to be "honest floors," not straw men**
The keyword-rule baseline was built to actually work on obvious cases (verified
against 9 hand-picked test cases before running for real) rather than being
deliberately weak -- so that its 20.5% result against the LLM's 75.0% is a fair,
defensible comparison rather than an inflated contrast.

**13. Report written in Markdown, not Word, for the primary deliverable**
Given the strict page-length constraint and iterative editing needs, chose
Markdown for the working report rather than docx -- easier to review and revise
quickly; can be converted to PDF/Word if the submission format requires it.
